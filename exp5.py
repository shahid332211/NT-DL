""" 5 Design and implement an Image classification model to classify a dataset of images using Deep Feed Forward Neural Network. Record t
accuracy corresponding to the number of epochs. Use the MNIST datasets."""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = Sequential([
Flatten(input_shape=(28, 28)),
Dense(128, activation='relu'),
Dense(256, activation='relu'),
Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')
