import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

IMG_SIZE = 64

data = []
labels = []

# Load open (label = 1)
for img in os.listdir("drowsy/dataset/Open_Eyes"):
    img_path = os.path.join("drowsy/dataset/Open_Eyes", img)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image)
    labels.append(1)

# Load closed (label = 0)
for img in os.listdir("drowsy/dataset/Closed_Eyes"):
    img_path = os.path.join("drowsy/dataset/Closed_Eyes", img)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image)
    labels.append(0)

data = np.array(data) / 255.0
labels = np.array(labels)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model.save("model/model.h5")

# Save test data for evaluation
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)