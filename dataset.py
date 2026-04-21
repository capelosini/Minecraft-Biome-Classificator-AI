from pathlib import Path

import keras
from keras import layers
from PIL import Image

image_size = (50, 50)
input_shape = image_size + (3,)
batch_size = 16

dataset_path = Path("dataset/")
class_names = sorted([f.name for f in dataset_path.iterdir() if f.is_dir()])
num_classes = len(list(dataset_path.iterdir()))

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def normalizeImage(img):
    img = img.crop(
        box=(
            (img.width / 2) - (img.height / 2),
            0,
            (img.width / 2) + (img.height / 2),
            img.height,
        )
    )
    return img


def normalizedOpenImage(imagePath, resize=None):
    img = Image.open(imagePath)
    img = normalizeImage(img)
    if resize:
        img = img.resize(resize)
    return img


def normalizeImagesRatioFiles():
    i = 0
    print("Normalizing images ratio...")
    for folder in dataset_path.iterdir():
        for imagePath in folder.iterdir():
            img = normalizedOpenImage(imagePath)
            img.save(imagePath)
            print(f"Normalized image {i}", end="\r")
            i += 1


train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="both",
    seed=1337,
    label_mode="categorical",
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# normalizeImagesRatio()
