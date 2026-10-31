
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "AgroVision_Final.keras")
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts", "all_v1")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

model.save(os.path.join(OUTPUT_DIR, "model.keras"))

print("Model exported successfully!")

# Save model metadata
metadata = {
    "model_name": "AgroVision",
    "framework": "TensorFlow/Keras",
    "architecture": "MobileNetV2",
    "num_classes": 38,
    "input_shape": [224, 224, 3],
    "classes": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
}

with open(os.path.join(OUTPUT_DIR, "artifact.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Metadata exported successfully!")

# Load validation portion of dataset
val_full_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = val_full_ds.class_names

val_batches = int(
    0.5 * tf.data.experimental.cardinality(val_full_ds).numpy()
)

val_ds = val_full_ds.take(val_batches)
test_ds = val_full_ds.skip(val_batches)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_ds)

metrics = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss)
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Metrics saved successfully!")

# Generate confusion matrices
def generate_confusion_matrix(dataset, filename, title):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    fig, ax = plt.subplots(figsize=(22, 22))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    plt.title(title)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(filename, "saved successfully!")


generate_confusion_matrix(
    val_ds,
    "confusion_val.png",
    "AgroVision Validation Confusion Matrix"
)

generate_confusion_matrix(
    test_ds,
    "confusion_test_in_domain.png",
    "AgroVision Test Confusion Matrix"
)

print("\nAll AgroVision artifacts generated successfully!")
