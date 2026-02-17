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

# ---------- TensorFlow / Keras imports ----------
# tensorflow (tf)  -- Google's deep-learning framework.  It handles tensor
#                     operations, automatic differentiation, and GPU acceleration.
# keras             -- a high-level API *inside* TensorFlow that lets you build
#                     and train models in just a few lines.  Think of it as
#                     "TensorFlow made simple".
# layers            -- pre-built building blocks: Conv2D, Dense, Dropout, etc.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ─── Load MNIST data ───────────────────────────────────────────────
# keras.datasets.mnist.load_data()
#   Downloads (or loads from cache) the classic MNIST dataset:
#     - 60,000 training images + labels
#     - 10,000 test images + labels
#   Each image is a 28x28 grayscale picture of a handwritten digit (0--9).
#   Pixel values are integers 0--255.
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ---- Reshape to 4-D: [samples, height, width, channels] ----
# The raw data has shape (60000, 28, 28) -- 3-D.
# Conv2D expects a 4-D tensor:  (batch, height, width, channels)
#   channels = 1 for grayscale, 3 for RGB colour images.
#   -1 in reshape means "infer this dimension automatically".
#
# ---- Normalize to [0, 1] ----
# Neural networks train faster and more stably when inputs are small.
# Dividing by 255.0 scales pixel values from [0, 255] to [0.0, 1.0].
# .astype("float32") ensures we use 32-bit floats (what TF expects).
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ---- One-hot encoding ----
# keras.utils.to_categorical(labels, num_classes)
#   Converts an integer label like 3 into a vector of length 10:
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#   This is required because our output layer uses softmax with 10 neurons
#   and categorical_crossentropy loss, which expects probability vectors.
#   Alternative: keep integer labels and use 'sparse_categorical_crossentropy'.
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

# ─── Build the CNN model ──────────────────────────────────────────
#
# keras.Sequential([...])
#   Creates a model where layers are stacked one after another in order.
#   Data flows from the first layer to the last automatically.
#   For more complex architectures (skip connections, multiple inputs)
#   you would use the Functional API instead.

model = keras.Sequential([

    # -------- CONV BLOCK 1 ------------------------------------------------
    #
    # layers.Conv2D(filters, kernel_size, activation, padding, input_shape)
    #
    #   filters (32) -- number of learned feature detectors.
    #       Each filter slides over the image and produces one output "channel".
    #       Early layers typically use fewer filters (edges, corners);
    #       deeper layers use more (complex patterns).
    #       Try 16 or 64 to see how capacity affects accuracy.
    #
    #   kernel_size (5, 5) -- the filter's height and width in pixels.
    #       Common choices: 3x3 (most popular), 5x5 (wider field of view).
    #       Larger kernels capture broader patterns but have more parameters.
    #
    #   activation='relu' -- Rectified Linear Unit: f(x) = max(0, x)
    #       Introduces non-linearity (without it, stacking layers would
    #       just be one big linear operation).  ReLU is the default choice
    #       for hidden layers because it is fast and avoids vanishing gradients.
    #
    #   padding='same' -- zero-pads the input so that the output has the
    #       SAME spatial dimensions as the input (28x28 in -> 28x28 out).
    #       Without padding (padding='valid'), a 5x5 kernel on 28x28 would
    #       produce 24x24 — we would lose 2 pixels on each edge.
    #
    #   input_shape=(28, 28, 1) -- only needed on the FIRST layer.
    #       Tells Keras the dimensions of each input image: 28 high, 28 wide,
    #       1 channel (grayscale).  Subsequent layers infer shapes automatically.
    #
    #   Trainable parameters: (5 * 5 * 1 + 1) * 32 = 832
    #     ↑ kernel_h * kernel_w * in_channels + 1 bias, times num_filters
    layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                  input_shape=(28, 28, 1)),

    # layers.MaxPooling2D(pool_size)
    #   Divides the feature map into non-overlapping 2x2 windows and keeps
    #   only the MAXIMUM value in each window.
    #   Effect: spatial dimensions are halved  (28x28 -> 14x14)
    #   Purpose:
    #     - Reduces computation for subsequent layers
    #     - Provides a degree of translation invariance (small shifts in
    #       the input don't change the pooled output much)
    #     - Forces the network to focus on the strongest activations
    #   Alternative: layers.AveragePooling2D (takes the mean instead)
    #   No trainable parameters.
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Output shape after block 1: (batch, 14, 14, 32)

    # -------- CONV BLOCK 2 ------------------------------------------------
    # Same idea, but with 64 filters -- capacity doubles so the network
    # can represent more complex patterns (curves, strokes, loops).
    # Input is now 14x14x32 (from block 1), so parameter count is:
    #   (5 * 5 * 32 + 1) * 64 = 51,264
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Output shape after block 2: (batch, 7, 7, 64)

    # -------- CLASSIFIER HEAD ---------------------------------------------
    #
    # layers.Flatten()
    #   Reshapes the 3-D feature map (7, 7, 64) into a 1-D vector of
    #   length 7 * 7 * 64 = 3,136.  This is necessary because Dense layers
    #   expect 1-D input.  No trainable parameters.
    layers.Flatten(),

    # layers.Dense(units, activation)
    #   A fully connected layer: every input neuron connects to every output.
    #   units=1024 -- number of neurons.  This is the "thinking" layer that
    #     learns non-linear combinations of the extracted features.
    #   Parameters: 3,136 * 1,024 + 1,024 biases = 3,212,288  (most of the model!)
    #   Insight: this single layer contains ~98% of the model's parameters.
    #     Reducing it (e.g. 256 units) dramatically shrinks the model.
    layers.Dense(1024, activation='relu'),

    # layers.Dropout(rate)
    #   During TRAINING, randomly sets `rate` fraction of inputs to 0.
    #   rate=0.5 means 50% of the 1,024 neurons are silenced each batch.
    #   Purpose: prevents the Dense layer from memorising the training data
    #     (overfitting).  Forces the network to learn redundant, robust features.
    #   During INFERENCE (model.predict / model.evaluate), Dropout is
    #     automatically disabled — all neurons are active.
    #   No trainable parameters.
    layers.Dropout(0.5),

    # layers.Dense(10, activation='softmax')
    #   Output layer with 10 neurons — one per digit class (0--9).
    #   softmax converts raw logits into a probability distribution that
    #   sums to 1.0.  The predicted class is the neuron with the highest
    #   probability: np.argmax(output).
    #   Parameters: 1,024 * 10 + 10 biases = 10,250
    layers.Dense(10, activation='softmax')
])

# model.summary() prints a table showing each layer's output shape and
# parameter count.  Very useful for debugging shape mismatches and
# understanding where most parameters live.
model.summary()

# ─── Compile the model ────────────────────────────────────────────
# model.compile() configures the model for training.  It does NOT train
# the model -- it just sets up the machinery (optimiser, loss, metrics).
#
# ---- optimizer: keras.optimizers.Adam(learning_rate) ----
#   Adam (Adaptive Moment Estimation) -- same algorithm used in the RNN
#   script.  It adapts the learning rate per-parameter and uses momentum.
#
#   learning_rate = 1e-4 (0.0001)
#     This is conservatively small.  Effects of changing it:
#       - Larger (e.g. 1e-3): faster training, but may overshoot minima
#       - Smaller (e.g. 1e-5): very stable but painfully slow
#     The default for Adam is 1e-3; we use 1e-4 because the Dense layer
#     has 3.2M parameters and a smaller LR prevents wild early updates.
#
#   Other optimisers you might encounter:
#     keras.optimizers.SGD(lr, momentum)  -- basic but effective with tuning
#     keras.optimizers.RMSprop(lr)        -- good for RNNs
#
# ---- loss: 'categorical_crossentropy' ----
#   The standard loss for multi-class classification with one-hot labels.
#   It measures how far the model's predicted probability distribution is
#   from the true one-hot vector.  Lower = better.
#   Formula per sample: -sum( y_true * log(y_pred) )
#   If you kept integer labels (not one-hot), use
#   'sparse_categorical_crossentropy' instead -- mathematically identical.
#
# ---- metrics: ['accuracy'] ----
#   Keras tracks these during training and displays them in the progress bar.
#   Accuracy = fraction of samples where argmax(prediction) == true label.
#   You can add more, e.g. metrics=['accuracy', 'AUC'].
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── Train the model ──────────────────────────────────────────────
# The original chapter21.ipynb talks about "iterations" rather than "epochs".
# An iteration = one gradient update on a single mini-batch.
# An epoch     = one full pass through the entire training set.
# Relationship:  iterations_per_epoch = num_samples / batch_size
num_iterations = 2000
batch_size = 75

# Convert iterations to epochs so we can use model.fit():
#   2000 iterations * 75 samples/batch / 60000 total samples = 2.5 epochs
#   We round down to 2 (max(1, ...) ensures at least 1 epoch).
epochs = max(1, (num_iterations * batch_size) // len(x_train))

print(f"\nTraining the model for {epochs} epochs "
      f"(~{num_iterations} iterations with batch size {batch_size})...\n")

# ---- model.fit() -- the main training loop ----
# Parameters:
#   x_train, y_train -- training data and labels.
#
#   batch_size (75) -- number of samples per gradient update.
#       - Smaller (e.g. 16):  noisier gradients, can help escape local minima,
#                             but slower (more updates per epoch).
#       - Larger  (e.g. 256): smoother gradients, faster per epoch, but may
#                             generalise worse and needs more memory.
#       75 is a moderate choice for MNIST.
#
#   epochs (2) -- how many full passes through the training data.
#       More epochs = more training.  Watch for overfitting: if train accuracy
#       keeps rising but validation accuracy plateaus or drops, you are
#       training too long.  (Dropout helps mitigate this.)
#
#   validation_split (0.1) -- reserves 10% of training data for validation.
#       Keras automatically holds out the LAST 10% of x_train/y_train
#       after each epoch to compute val_loss and val_accuracy.
#       This lets you monitor overfitting without touching the test set.
#       Note: the data is NOT shuffled before splitting, so the last 6,000
#       samples become the validation set.
#
#   verbose (1) -- progress bar during training.  0 = silent, 2 = one line/epoch.
#
# Returns a History object (we ignore it here) that contains loss/accuracy
# values for every epoch -- useful for plotting learning curves.
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1
)

# ─── Evaluate on test data ────────────────────────────────────────
# model.evaluate(x, y)
#   Runs the model on the test set (forward pass only, no training)
#   and returns the values of the loss and any metrics specified in compile().
#   verbose=0 suppresses the progress bar.
#
#   IMPORTANT: the test set must NEVER be used during training or
#   hyper-parameter tuning.  It is the final, unbiased measure of how
#   well the model generalises to completely unseen data.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy = {test_accuracy:.4f}")
# With this architecture and ~2.5 epochs you can expect ~98% accuracy.
# Training for more epochs (e.g. 10-20) typically pushes it above 99%.

# ─── Visualize Predictions ────────────────────────────────────────
# Select 10 random test images to inspect visually.
indices = np.random.choice(len(x_test), 10, replace=False)
x_sample = x_test[indices]
y_sample_true = y_test[indices]

# model.predict(x)
#   Runs the forward pass and returns the output of the LAST layer.
#   For our model that is a (10,) softmax vector per sample -- each
#   element is the predicted probability of the corresponding digit.
print("\nGenerating predictions for visualization...")
y_sample_pred = model.predict(x_sample)

# np.argmax(array, axis=1)
#   Returns the INDEX of the maximum value along axis 1 (columns).
#   Since the softmax outputs probabilities for digits 0--9, argmax
#   gives us the predicted class label (the digit with highest probability).
pred_labels = np.argmax(y_sample_pred, axis=1)
true_labels = np.argmax(y_sample_true, axis=1)

# ---- Plot the 10 samples with predictions ----
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    # Green title = correct prediction, Red = wrong
    color = 'green' if pred_labels[i] == true_labels[i] else 'red'
    plt.title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}", color=color)
    plt.axis('off')

plt.suptitle("CNN Predictions on Random Test Samples")
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────
# KEY TAKEAWAYS
# ──────────────────────────────────────────────────────────────────
#
# a) WHY CNNs for images?
#    A Dense network on 28x28 images would need 784 inputs per neuron and
#    would ignore spatial structure entirely.  Conv2D exploits locality:
#    each filter only looks at a small patch (5x5), reuses the same weights
#    across the whole image ("weight sharing"), and stacks features
#    hierarchically (edges -> textures -> parts -> objects).
#
# b) PARAMETER EFFICIENCY
#    The two Conv2D layers have only ~52 K parameters but extract rich
#    spatial features.  Most parameters (3.2 M) live in the Dense(1024)
#    layer -- that's a common bottleneck in CNN design.
#
# c) REGULARISATION
#    Dropout(0.5) is the main defence against overfitting here.  Other
#    techniques you might add:
#      - Data augmentation (random rotations, shifts)
#      - Weight decay (L2 regularisation in the optimiser)
#      - Early stopping (stop training when val_loss stops improving)
#
# d) GOING FURTHER
#    - Replace the 5x5 kernels with two stacked 3x3 kernels (same receptive
#      field, fewer parameters -- this is the VGG idea).
#    - Add BatchNormalization after each Conv2D for faster, more stable training.
#    - Try a deeper architecture (3-4 conv blocks) and see if accuracy improves.
#    - Use nn.LSTM or nn.GRU from Chapter 21 on the SAME data -- treat each
#      row of the 28x28 image as a time step.  Compare performance with CNN.
