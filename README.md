# Minecraft Biome Classification Project

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![PyTorch Backend](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Arch](https://img.shields.io/badge/Arch%20Linux-1793D1?logo=arch-linux&logoColor=fff&style=for-the-badge)
![KDE Plasma](https://img.shields.io/badge/KDE%20Plasma-1D99F3?logo=kdeplasma&logoColor=fff&style=for-the-badge)

---

## 1. Dataset Preparation
The original dataset required significant interventions to ensure training quality:

* **Noise Handling:** Since the images were extracted from recordings, many contained visual noise. Manual cleaning was performed to remove frames where the view was obstructed (e.g., character on fire or facing walls).
* **Classification:** Manual reclassification of biomes was conducted to ensure label accuracy.
* **Balancing:** To address the disparity in the number of images per class, the `prune_images` function was used to randomly remove captures from over-represented biomes.
* **Standardization:** Images were processed by the `predict_image` function, which resizes them to a **50x50** pixel format.
* **Source:** The dataset was obtained via Kaggle: [Minecraft Biomes](https://www.kaggle.com/datasets/willowc/minecraft-biomes).

---

## 2. Dataset Composition
The model works with the following biome categories:
* Badlands
* Desert
* Forest
* Plains
* Savanna
* Snow
* Swamp
* Taiga

---

## 3. Model Training
The chosen architecture was **VGG19**, utilizing pre-trained **ImageNet** weights as the convolutional base for feature extraction.

### Training Strategy
Training was structured in two stages:
1. **Initial Training:** Freezing the VGG19 weights and training only the top dense layers (the classifier).
2. **Fine-tuning:** Unfreezing the last 4 layers of the pre-trained base for fine adjustments to the specific Minecraft data.

### Hyperparameters and Settings
* **Optimizer:** Adam with a learning rate of $1 \times 10^{-5}$.
* **Cycles:** 20 epochs for initial training and 10 epochs for fine-tuning.
* **Validation:** Use of `validation_data=val_ds` to test accuracy on images not seen during training.
* **Persistence:** The model is saved automatically via `save_best_only=True`, always maintaining the best state achieved.

### Data Augmentation
To increase data diversity and prevent overfitting, we applied:
* **RandomZoom:** 10% random zoom.
* **RandomContrast:** Random contrast adjustments.
* **RandomTranslation:** Horizontal and vertical shifts.

---

## 4. Model Usage
The model processes new images and returns an array of probabilities for each biome.

* **Operation:** The AI analyzes the frame and indicates the probability of that image belonging to each class.
* **Real-Time Interface:** If used during a live stream, a side display shows the player which biome the AI identified and the respective confidence percentage.

---

## 5. Model Testing
To validate the effectiveness of the solution, the following procedures were performed:
* **Test Dataset:** Application of evaluation metrics (Accuracy) on the separate test dataset.
* **Unit Tests:** Individual testing with images external to the original dataset to verify the AI's generalization capability across different gameplay scenarios.

---

## 6. Model Testing in live



https://github.com/user-attachments/assets/b73d278e-cac8-4790-894c-f1a0bf06d4ac




