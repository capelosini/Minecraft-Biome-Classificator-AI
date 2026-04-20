from pathlib import Path

import keras
from PIL import Image

image_size = (50, 50)
batch_size = 128

datasetFolder = Path("dataset/")


def normalizeImagesRatio():
    i = 0
    print("Normalizing images ratio...")
    for folder in datasetFolder.iterdir():
        for imagePath in folder.iterdir():
            img = Image.open(imagePath)
            img = img.crop(
                box=(
                    (img.width / 2) - (img.height / 2),
                    0,
                    (img.width / 2) + (img.height / 2),
                    img.height,
                )
            )
            img.save(imagePath)
            print(f"Normalized image {i}", end="\r")
            i += 1


train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# normalizeImagesRatio()
