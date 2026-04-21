from pathlib import Path

import keras
import numpy as np

from dataset import (
    class_names,
    image_size,
    input_shape,
    normalizedOpenImage,
    num_classes,
)
from model import make_model

model = make_model(input_shape, num_classes)
model.trainable = False
model.build((1,) + input_shape)

model.load_weights("checkpoints/model_finetuned.keras")

model.summary()


def predict_image(img_path):
    # Load and resize to match your training (50x50)
    img = normalizedOpenImage(img_path, resize=image_size)

    # Convert to array and add a 'batch' dimension: (50, 50, 3) -> (1, 50, 50, 3)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]

    # Get highest probability index
    class_idx = np.argmax(score)
    confidence = 100 * np.max(score)

    print(f"Image: {Path(img_path).name}")
    print(f"Predicted: {class_names[class_idx]} ({confidence:.2f}% confidence)")
    print("-" * 30)


if __name__ == "__main__":
    predict_image("demoDataset/swamp1.webp")
