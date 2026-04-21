import os

# This prevents fragmentation and allows PyTorch to expand segments more efficiently
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import keras
import torch

from dataset import data_augmentation_layers, input_shape, num_classes, train_ds, val_ds
from model import make_model

# Training
model = make_model(input_shape, num_classes)
epochs = 10

callbacks = [
    keras.callbacks.ModelCheckpoint("checkpoints/model.keras", save_best_only=True),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

torch.cuda.empty_cache()

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# finetuning
print("Switching to Fine-Tuning...")
# Unfreeze the base
base = model.layers[1]
base.trainable = True

# Refreeze everything EXCEPT the last block
for layer in base.layers[:-4]:
    layer.trainable = False

# Recompile with a MUCH smaller learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # 10x or 100x smaller
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

torch.cuda.empty_cache()

fineTuningCallbacks = [
    keras.callbacks.ModelCheckpoint(
        "checkpoints/model_finetuned.keras", save_best_only=True
    ),
]

# Extend augmentation
# Adding more noise for the fine-tuning phase
data_augmentation_layers.extend(
    [
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ]
)

model.fit(
    train_ds,
    epochs=5,
    callbacks=fineTuningCallbacks,
    validation_data=val_ds,
)
