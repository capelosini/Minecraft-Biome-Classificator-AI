from keras import Sequential, layers
from keras.applications import VGG19


def make_model(input_shape, num_classes):
    # Entry block
    base = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False

    model = Sequential(
        [
            layers.Rescaling(1.0 / 255),
            base,
            layers.GlobalAveragePooling2D(),  # Use this instead of Flatten()
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
