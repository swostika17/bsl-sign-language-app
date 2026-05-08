import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ----------------------------
# Create folders if missing
# ----------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("backend", exist_ok=True)


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("hand_landmarks.csv")

X = df.drop("label", axis=1).values
y = df["label"].values


# ----------------------------
# Encode labels
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)


# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ----------------------------
# Build Model
# ----------------------------
model = Sequential([
    Dense(512, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# ----------------------------
# Evaluate Model
# ----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print("\nOverall Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ----------------------------
# Plot Confusion Matrix
# ----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ----------------------------
# Plot Accuracy Graph
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_graph.png", dpi=300)
plt.show()


# ----------------------------
# Plot Loss Graph
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_graph.png", dpi=300)
plt.show()


# ----------------------------
# Save Model and Labels
# ----------------------------
model.save("model/sign_model.h5")
np.save("model/labels.npy", le.classes_)

model.save("backend/sign_model.h5")
np.save("backend/labels.npy", le.classes_)

print("\nModel trained and saved successfully.")
print("Saved files:")
print("- model/sign_model.h5")
print("- model/labels.npy")
print("- backend/sign_model.h5")
print("- backend/labels.npy")
print("- confusion_matrix.png")
print("- accuracy_graph.png")
print("- loss_graph.png")