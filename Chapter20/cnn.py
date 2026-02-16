"""
CNN for MNIST digit classification
===================================
Builds a Convolutional Neural Network using TensorFlow/Keras
to classify handwritten digits from the MNIST dataset.

Architecture (same as the original):
  Conv2D(32, 5x5) -> MaxPool(2x2) ->
  Conv2D(64, 5x5) -> MaxPool(2x2) ->
  Flatten -> Dense(1024) -> Dropout(0.5) -> Dense(10)

Usage:
  python cnn.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ─── Load MNIST data ───────────────────────────────────────────────
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape to [samples, 28, 28, 1] and normalize to [0, 1]
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

# ─── Build the CNN model ──────────────────────────────────────────
model = keras.Sequential([
    # First convolutional layer: 32 filters, 5x5 kernel, ReLU
    layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional layer: 64 filters, 5x5 kernel, ReLU
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten and fully connected layer
    layers.Flatten(),                       # 7 * 7 * 64 = 3136
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),

    # Output layer: 10 classes (digits 0-9)
    layers.Dense(10, activation='softmax')
])

model.summary()

# ─── Compile the model ────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── Train the model ──────────────────────────────────────────────
num_iterations = 2000
batch_size = 75

# Calculate equivalent epochs: 2000 iterations * 75 batch / 60000 samples ≈ 2.5 epochs
epochs = max(1, (num_iterations * batch_size) // len(x_train))

print(f"\nTraining the model for {epochs} epochs "
      f"(~{num_iterations} iterations with batch size {batch_size})...\n")

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1
)

# ─── Evaluate on test data ────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy = {test_accuracy:.4f}")

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
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    color = 'green' if pred_labels[i] == true_labels[i] else 'red'
    plt.title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}", color=color)
    plt.axis('off')

plt.suptitle("CNN Predictions on Random Test Samples")
plt.tight_layout()
plt.show()
