from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="BSL Sign Language API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    print("Loading model...")

    model = load_model("sign_model.h5", compile=False)
    labels = np.load("labels.npy", allow_pickle=True)

    print("Model loaded successfully")
    print("Labels:", labels)

except Exception as e:
    print("Error loading model:", e)
    raise e


class Features(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "BSL Sign Language API is running"}


@app.post("/predict")
def predict(data: Features):
    features = data.features

    if len(features) != 126:
        raise HTTPException(
            status_code=400,
            detail=f"Feature vector must contain exactly 126 values. Received {len(features)}"
        )

    try:
        X = np.array(features, dtype=np.float32).reshape(1, -1)

        probs = model.predict(X, verbose=0)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = str(labels[idx])

        if confidence < 0.60:
            return {
                "prediction": "Uncertain",
                "confidence": confidence
            }

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )