# This live script will only work in linux kde plasma

import subprocess
import tkinter as tk
from threading import Thread

import keras
import numpy as np

from dataset import class_names, input_shape, normalizedOpenImage, num_classes
from model import make_model

model = make_model(input_shape, num_classes)
model.trainable = False
model.build((1,) + input_shape)

model.load_weights("checkpoints/model_finetuned.keras")

model.summary()


def predict_image():
    subprocess.run(["spectacle", "-a", "-b", "-o", "window.png"])

    img = normalizedOpenImage("window.png", resize=(50, 50))

    # Convert to array and add a 'batch' dimension: (50, 50, 3) -> (1, 50, 50, 3)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    print("predicting...")
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]

    # Get highest probability index
    # class_idx = np.argmax(score)
    # confidence = 100 * np.max(score)
    class_name = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return f"{class_name}: {confidence:.4f}%"


root = tk.Tk()

# 1. Keep the window on top of all others
root.attributes("-topmost", True)

# 2. (Optional) Remove window borders and title bar for a clean overlay look
root.overrideredirect(True)

# 3. (Optional) Set window transparency (0.0 to 1.0)
root.attributes("-alpha", 0.8)

classVar = tk.StringVar()
classVar.set("")

# Add your text
label = tk.Label(root, textvariable=classVar, font=("Arial", 20), fg="red", bg="white")
label.pack()

# Position the window (widthxheight+x+y)
root.geometry("450x100+100+100")


def update_label():
    result = predict_image()
    classVar.set(result)
    root.after(1000, update_label)


Thread(target=update_label).start()

root.mainloop()
