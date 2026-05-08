speechSynthesis.onvoiceschanged = () => {
    console.log("Voices loaded:", speechSynthesis.getVoices().map(v => v.name));
};

const BSL_TRANSLATIONS = {
    "How are you?": "How are you?",
    "Nice to meet you": "Nice to meet you",
    "Ok": "OK",
    "Sorry": "Sorry",
    "Thank you": "Thank you",
    "hello": "Hello",
    "peace": "Peace"
};

let hands = null;
let stableCount = 0;
let lastLabel = "";
let lastSpokenLabel = "";
let isSpeaking = false;

const STABLE_FRAMES = 4;
const CONF_THRESHOLD = 0.60;

const welcomeScreen = document.getElementById("welcomeScreen");
const mainApp = document.getElementById("mainApp");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const signDisplay = document.getElementById("signDisplay");
const textTranslation = document.getElementById("textTranslation");
const handsCountEl = document.getElementById("handsCount");
const confidenceEl = document.getElementById("confidence");

function startApp() {
    welcomeScreen.style.display = "none";
    mainApp.style.display = "block";
    initHands();
}

function speakTranslation(text) {
    if (isSpeaking) return;

    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.85;
    utterance.pitch = 1.25;
    utterance.volume = 1;

    const voices = speechSynthesis.getVoices();

    const femaleVoice =
        voices.find(v => v.name.toLowerCase().includes("samantha")) ||
        voices.find(v => v.name.toLowerCase().includes("karen")) ||
        voices.find(v => v.name.toLowerCase().includes("moira")) ||
        voices.find(v => v.name.toLowerCase().includes("serena")) ||
        voices.find(v => v.lang === "en-GB") ||
        voices.find(v => v.lang.startsWith("en"));

    if (femaleVoice) {
        utterance.voice = femaleVoice;
        console.log("Female voice selected:", femaleVoice.name);
    }

    utterance.onstart = () => {
        isSpeaking = true;
    };

    utterance.onend = () => {
        isSpeaking = false;
    };

    speechSynthesis.speak(utterance);
}

function extractFeatures(results) {
    let allHands = results.multiHandLandmarks || [];

    allHands.sort((h1, h2) => {
        const cx1 = h1.reduce((sum, lm) => sum + lm.x, 0) / 21;
        const cx2 = h2.reduce((sum, lm) => sum + lm.x, 0) / 21;
        return cx1 - cx2;
    });

    let features = [];

    for (let i = 0; i < Math.min(2, allHands.length); i++) {
        let hand = allHands[i];

        let lm = hand.map(p => [p.x, p.y, p.z]);
        let wrist = lm[0];

        lm = lm.map(p => [
            p[0] - wrist[0],
            p[1] - wrist[1],
            p[2] - wrist[2]
        ]);

        let lm9 = lm[9];
        let scale = Math.sqrt(
            lm9[0] * lm9[0] +
            lm9[1] * lm9[1] +
            lm9[2] * lm9[2]
        ) + 1e-6;

        lm = lm.map(p => [
            p[0] / scale,
            p[1] / scale,
            p[2] / scale
        ]);

        features.push(...lm.flat());
    }

    while (features.length < 126) {
        features.push(0);
    }

    return features.slice(0, 126);
}

async function predictBackend(features) {
    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ features: features })
        });

        if (!res.ok) {
            throw new Error("Backend error");
        }

        const data = await res.json();

        if (data.prediction === "Uncertain" || data.confidence < CONF_THRESHOLD) {
            stableCount = 0;
            signDisplay.textContent = "👐";
            textTranslation.textContent = "Please show a clear BSL sign";
            confidenceEl.textContent = `${Math.round(data.confidence * 100)}%`;
            return;
        }

        if (data.prediction !== lastLabel) {
            lastLabel = data.prediction;
            stableCount = 0;
        }

        stableCount++;

        if (stableCount >= STABLE_FRAMES) {
            const english = BSL_TRANSLATIONS[data.prediction] || data.prediction;

            signDisplay.textContent = data.prediction.toUpperCase();
            textTranslation.textContent = `"${english}"`;
            confidenceEl.textContent = `${Math.round(data.confidence * 100)}%`;

            if (data.prediction !== lastSpokenLabel && !isSpeaking) {
                speakTranslation(english);
                lastSpokenLabel = data.prediction;
            }
        }

    } catch (error) {
        console.error("Backend connection error:", error);
        signDisplay.textContent = "🔌";
        textTranslation.textContent = "Backend not connected";
        confidenceEl.textContent = "0%";
    }
}

function onResults(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        handsCountEl.textContent = results.multiHandLandmarks.length;

        for (let landmarks of results.multiHandLandmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                color: "#00E5FF",
                lineWidth: 4
            });

            drawLandmarks(ctx, landmarks, {
                color: "#FFD700",
                radius: 6
            });
        }

        const features = extractFeatures(results);
        predictBackend(features);

    } else {
        handsCountEl.textContent = "0";
        stableCount = 0;
        signDisplay.textContent = "👐";
        textTranslation.textContent = "Show your BSL sign";
        confidenceEl.textContent = "0%";
    }
}

function initHands() {
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);

    const camera = new Camera(video, {
        onFrame: async () => {
            await hands.send({ image: video });
        },
        width: 640,
        height: 480
    });

    camera.start();

    canvas.width = 640;
    canvas.height = 480;
}