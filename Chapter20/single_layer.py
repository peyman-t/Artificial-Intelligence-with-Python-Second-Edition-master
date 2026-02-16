"""
Single-layer neural network for MNIST classification
=====================================================
A simple single-layer (logistic regression) model using
TensorFlow/Keras to classify handwritten digits.

Architecture: Input(784) -> Dense(10, softmax)

Usage:
  python single_layer.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Get the MNIST data
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# The images are 28x28, so flatten to 784 and normalize to [0, 1]
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

# Convert labels to one-hot encoding (10 distinct digits)
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

# Create a single-layer model with weights and biases
# There are 10 distinct digits, so the output layer has 10 classes
model = keras.Sequential([
    layers.Dense(10, activation='softmax', input_shape=(784,))
])

model.summary()

# Define the loss and the gradient descent optimizer
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Start training
num_iterations = 1200
batch_size = 90

# Calculate equivalent epochs: 1200 iterations * 90 batch / 60000 samples ≈ 1.8 epochs
epochs = max(1, (num_iterations * batch_size) // len(x_train))

print(f"\nTraining the model for {epochs} epoch(s) "
      f"(~{num_iterations} iterations with batch size {batch_size})...\n")

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

# Compute the accuracy using test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAccuracy = {test_accuracy:.4f}")


# ─── Visualize Predictions ────────────────────────────────────────
# Select 10 random test images
indices = np.random.choice(len(x_test), 10, replace=False)
x_sample = x_test[indices]
y_sample_true = y_test[indices]

# Predict
print("\nGenerating predictions for visualization...")
y_sample_pred = model.predict(x_sample)
pred_labels = np.argmax(y_sample_pred, axis=1)
true_labels = np.argmax(y_sample_true, axis=1)

# Plot
plt.figure(figsize=(15, 3))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    # Reshape flattened 784 -> 28x28 for display
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    color = 'green' if pred_labels[i] == true_labels[i] else 'red'
    plt.title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}", color=color)
    plt.axis('off')

plt.suptitle("Single Layer Model Predictions")
plt.tight_layout()
plt.show()


