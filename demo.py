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
    # class_idx = np.argmax(score)
    # confidence = 100 * np.max(score)

    print(f"\nImage: {Path(img_path).name}\n")
    scores = {}
    for i in range(len(score)):
        scores[class_names[i]] = score[i]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for class_name, score in sorted_scores:
        print(f"{class_name}: {score * 100:.4f}%")
    print("-" * 30)


if __name__ == "__main__":
    demoDataset = Path("demoDataset")

    for imgPath in demoDataset.iterdir():
        predict_image(imgPath)
