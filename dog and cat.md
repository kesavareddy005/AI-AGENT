# 🐱🐶 Cat vs Dog Image Classification Project

This repository presents a deep learning project that classifies images of cats and dogs using convolutional neural networks (CNNs) with TensorFlow and Keras.

---

## 📁 1. Project Content

This project includes:

- **Data Acquisition**: Downloading and organizing images of cats and dogs.
- **Data Preprocessing**: Image resizing, augmentation, and scaling.
- **Model Building**: Defining a CNN architecture using Keras.
- **Model Training**: Optimizing the model using Binary Cross-Entropy loss.
- **Evaluation & Visualization**: Visualizing loss and accuracy trends, as well as example predictions.

The complete code is contained in the Jupyter notebook `cat_dog (1).ipynb`.

---

## 💻 2. Project Code

Here’s a quick glimpse into the core code components of the project:

### Data Mounting & Path Setup
```python
from google.colab import drive
drive.mount('/content/drive')

train_path = '/content/drive/MyDrive/cat_dog'
 
---

Key Library Imports
python
Copy
Edit
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    LeakyReLU, BatchNormalization, InputLayer, RandomFlip
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
Dataset Download
python
Copy
Edit
!pip install gdown
import gdown

gdown.download('https://drive.google.com/uc?id=1pIdmLUSqocdVCJTHjks_lZF3WB_9g5QI', 'cat_and_dog.zip', quiet=False)
Model Definition
python
Copy
Edit
model = Sequential([
    InputLayer(input_shape=(150, 150, 3)),
    RandomFlip("horizontal"),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])
Training
python
Copy
Edit
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
Evaluation
python
Copy
Edit
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
