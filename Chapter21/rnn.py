"""
============================================================
 Simple RNN Tutorial – Predict the Next Value of a Sine Wave
============================================================

Difficulty : ★☆☆☆☆  (introductory)
Concepts   : sequential data, hidden state, RNN cell, time-series prediction
Framework  : PyTorch (uses nn.RNN so students focus on *concepts*, not manual math)

Comparison with chapter21.ipynb
-------------------------------
The notebook builds an RNN **entirely from scratch** with NumPy — 130+ lines of
manual forward passes, BPTT, and gradient clipping.  That is great for deep
understanding, but overwhelming as a first exposure.

This script uses PyTorch's built-in RNN layer so you can:
  1. See the big picture first  (data → model → train → predict)
  2. Focus on *what* an RNN does, not *how* to code the math
  3. Experiment quickly (change hidden size, sequence length, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- PyTorch imports ----------
# torch          – the core tensor library (like NumPy, but with GPU support
#                  and automatic differentiation for training neural networks).
# torch.nn (nn)  – contains all the building blocks for neural networks:
#                  layers (Linear, RNN, Conv2d ...), loss functions (MSELoss,
#                  CrossEntropyLoss ...), and the base class nn.Module that
#                  every custom model inherits from.
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# 0.  SEED for reproducibility
# ──────────────────────────────────────────────────────────
# torch.manual_seed(seed)
#   Sets the random-number generator seed for ALL CPU tensor operations
#   (weight initialisation, dropout masks, etc.).  Using a fixed seed means
#   you will get the *exact same* results every time you run the script,
#   which is essential for debugging and for comparing experiments fairly.
#   If you also use a GPU, call  torch.cuda.manual_seed_all(42)  as well.
torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────
# 1.  GENERATE DATA – a simple sine wave
# ──────────────────────────────────────────────────────────
#   We sample 300 points from sin(x).  Each training example is a short
#   window of consecutive values, and the label is the *next* value.
#
#   Example (window_size=5):
#       Input :  [sin(0), sin(1), sin(2), sin(3), sin(4)]
#       Target:   sin(5)

# np.linspace(start, stop, num) creates `num` evenly-spaced values in [start, stop].
# We go from 0 to 20*pi so we get ~10 full sine cycles (period of sin = 2*pi).
# .astype(np.float32) – PyTorch defaults to 32-bit floats; matching that here
#   avoids a silent dtype mismatch later when we create tensors.
raw = np.sin(np.linspace(0, 20 * np.pi, 300)).astype(np.float32)

# ---- Quick look at the data ----
# Always visualise your data before modelling!  For time-series tasks you want
# to confirm the signal is clean, periodic, and has enough samples per cycle.
plt.figure(figsize=(10, 2))
plt.title("Raw sine-wave data (300 points)")
plt.plot(raw)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────
# 2.  CREATE SEQUENCES  (sliding window)
# ──────────────────────────────────────────────────────────
# WINDOW = how many past time steps the RNN looks at to make one prediction.
# Think of it as the model's "short-term memory" length.
#   - Too small (e.g. 2-3): the model sees too little context and struggles.
#   - Too large (e.g. 100): training is slower and the model may overfit.
#   - A good rule of thumb: start with ~1 period of the signal (here ~30 pts
#     per cycle), then shrink it to see how little context you can get away with.
WINDOW = 10

def make_sequences(data, window):
    """Slide a window over `data`; return (inputs, targets) tensors.

    For a dataset [v0, v1, v2, ..., vN] with window=3 this produces:
        X[0] = [v0, v1, v2]   Y[0] = v3
        X[1] = [v1, v2, v3]   Y[1] = v4
        ...  and so on.
    """
    xs, ys = [], []
    for i in range(len(data) - window):
        xs.append(data[i : i + window])
        ys.append(data[i + window])          # next value after the window

    # ---- Convert Python lists to PyTorch tensors ----
    # torch.tensor(array)
    #   Creates a new tensor that *copies* the data from the NumPy array.
    #   The resulting tensor lives on CPU and has the same dtype (float32).
    #
    # .unsqueeze(dim)
    #   Inserts a new dimension of size 1 at position `dim`.
    #   We need this because nn.RNN expects input shape:
    #       (batch_size, sequence_length, input_size)
    #   Our raw X is (N, window) -- 2-D.  unsqueeze(-1) makes it (N, window, 1)
    #   where input_size=1 (one feature: the sine value at each time step).
    #
    #   If we had multiple features per step (e.g. temperature AND humidity),
    #   input_size would be 2, and we wouldn't need unsqueeze.
    X = torch.tensor(np.array(xs)).unsqueeze(-1)   # shape: (N, window, 1)
    Y = torch.tensor(np.array(ys)).unsqueeze(-1)   # shape: (N, 1)
    return X, Y

X, Y = make_sequences(raw, WINDOW)

# ---- Train / validation split ----
# We take the first 80% of sequences for training and the rest for validation.
# IMPORTANT: for time-series data we do NOT shuffle!  Shuffling would let the
# model "peek" at future data that comes after the validation window.
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val,   Y_val   = X[split:], Y[split:]

print(f"Training examples : {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Each input shape   : {tuple(X_train[0].shape)}  (window x features)")

# ──────────────────────────────────────────────────────────
# 3.  WHAT HAPPENS INSIDE AN RNN CELL?  (conceptual demo)
# ──────────────────────────────────────────────────────────
#   Before we use PyTorch's nn.RNN, let's see the core equation:
#
#       h_t = tanh( W_ih · x_t  +  W_hh · h_{t-1}  +  bias )
#
#   • x_t      = input at time step t
#   • h_{t-1}  = hidden state from the *previous* time step
#   • h_t      = new hidden state (also the output at step t)
#   • W_ih, W_hh = learned weight matrices
#
#   The key insight: h carries a *memory* of all previous inputs.
#   After processing the whole sequence, the final h summarises
#   everything the RNN has "read".

print("\n--- Manual single-cell demo (1 sequence, 3 time-steps) ---")
input_size  = 1
hidden_size = 4     # tiny hidden state so the printout is readable

# torch.randn(rows, cols)
#   Creates a matrix filled with random values drawn from a standard normal
#   distribution (mean=0, std=1).  In a real model, PyTorch initialises
#   weights like this internally and then *learns* better values via backprop.
W_ih = torch.randn(hidden_size, input_size)     # maps input  -> hidden
W_hh = torch.randn(hidden_size, hidden_size)    # maps hidden -> hidden (recurrence!)
bias = torch.zeros(hidden_size)                  # bias term (starts at 0)

# torch.zeros(size)
#   Creates a tensor of all zeros.  The hidden state h_0 is conventionally
#   initialised to zeros -- the network has "no memory" before seeing any data.
h = torch.zeros(hidden_size)
demo_seq = X_train[0]                   # one training sequence, shape (WINDOW, 1)

# Step through the first 3 time steps to see how h evolves:
for t in range(3):
    x_t = demo_seq[t]                   # scalar input at this time step
    # ---------- THE CORE RNN EQUATION ----------
    #   h_t = tanh( W_ih * x_t  +  W_hh * h_{t-1}  +  bias )
    #
    # @ is Python's matrix-multiply operator (same as torch.matmul).
    #   W_ih @ x_t  -- project the current input into hidden space
    #   W_hh @ h    -- project the *previous* hidden state into hidden space
    # Adding them mixes "what I just saw" with "what I remember".
    #
    # torch.tanh squashes the result to [-1, +1], which:
    #   - prevents values from exploding over many time steps
    #   - introduces non-linearity (without it, stacking steps
    #     would just be one big linear transformation)
    h = torch.tanh(W_ih @ x_t + W_hh @ h + bias)
    print(f"  t={t}  x_t={x_t.item():.3f}  ->  h_t = {h.numpy().round(3)}")

print("  ... (h keeps updating at every time step)")
print("  Notice how h changes even when x_t values are similar --")
print("  that is because h also depends on its OWN previous value (the recurrence).\n")

# ──────────────────────────────────────────────────────────
# 4.  DEFINE THE MODEL
# ──────────────────────────────────────────────────────────
#   Architecture (intentionally tiny):
#       Input (1 feature per step)
#         → RNN layer  (hidden_size = 16)
#         → Linear     (16 → 1)        ← predicts next sine value
#
#   Total parameters: ~300  (vs. 3.2 M in the CNN from Ch 20!)

# ---- nn.Module: the base class for every PyTorch model ----
# By subclassing nn.Module you get:
#   - automatic parameter tracking  (model.parameters() returns all weights)
#   - .train() / .eval() mode switching  (affects Dropout, BatchNorm, etc.)
#   - easy saving/loading  (torch.save / torch.load)
#   - clean __repr__  (print(model) shows the architecture)
#
# You MUST implement two methods:
#   __init__  : define the layers
#   forward   : define how data flows through them

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()    # always call the parent constructor first

        # ---- nn.RNN(input_size, hidden_size, ...) ----
        # This creates a single-layer Elman RNN.
        #
        # Parameters explained:
        #   input_size   (int) -- number of features at each time step.
        #       We have 1 feature (the sine value).  For NLP tasks with
        #       word embeddings this might be 128 or 300.
        #
        #   hidden_size  (int) -- number of neurons in the hidden state.
        #       This is the "memory capacity" of the RNN.
        #       - Larger  -> can remember more complex patterns, but needs
        #                    more data and trains slower.
        #       - Smaller -> faster, but may under-fit.
        #       Try 4 vs 16 vs 64 to see the effect.
        #
        #   batch_first (bool, default False) -- controls tensor layout.
        #       True  -> input/output shape is (batch, seq_len, features)
        #       False -> shape is (seq_len, batch, features)
        #       batch_first=True is more intuitive and matches how we
        #       built our data, so we always set it.
        #
        # Other useful optional parameters (not used here):
        #   num_layers  (int, default 1)
        #       Stack multiple RNN layers. Layer 2's input is layer 1's output.
        #       Deeper = more capacity, but harder to train (vanishing gradients).
        #   nonlinearity (str, 'tanh' or 'relu', default 'tanh')
        #       Activation function applied at each step.
        #       'relu' can help with vanishing gradients but may cause exploding.
        #   dropout (float, default 0)
        #       Dropout rate between stacked RNN layers (requires num_layers > 1).
        #   bidirectional (bool, default False)
        #       If True, runs the sequence both forward and backward, doubling
        #       the hidden size.  Useful for NLP but not for causal time-series.
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # ---- nn.Linear(in_features, out_features) ----
        # A standard fully-connected (dense) layer:  output = x @ W^T + b
        #   in_features  = hidden_size (the RNN's output dimension)
        #   out_features = output_size (1, because we predict a single number)
        # This layer acts as a "decoder" that converts the RNN's hidden
        # representation into the final prediction.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)  e.g. (232, 10, 1)
        #
        # ---- What self.rnn(x) returns ----
        # rnn_out : (batch, seq_len, hidden_size)
        #     The hidden state at EVERY time step.  rnn_out[:, 0, :] is h_1,
        #     rnn_out[:, -1, :] is h_T (the last step).
        #
        # h_n     : (num_layers, batch, hidden_size)
        #     The hidden state at the LAST time step only.  For a 1-layer RNN,
        #     h_n[0] is identical to rnn_out[:, -1, :].  It is handy when you
        #     stack multiple layers -- h_n gives the final state of each layer.
        rnn_out, h_n = self.rnn(x)

        # We only care about the prediction after the entire sequence has been
        # processed, so we take the output at the LAST time step:
        last_hidden = rnn_out[:, -1, :]       # shape: (batch, hidden_size)

        # Pass through the linear layer to get 1 output value per sample:
        prediction = self.fc(last_hidden)     # shape: (batch, output_size)
        return prediction

# ---- Instantiate the model ----
model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)

# Count parameters: p.numel() returns the total number of elements in a tensor.
# For an RNN with hidden_size=16 and input_size=1:
#   W_ih: 16x1 = 16,  W_hh: 16x16 = 256,  bias_ih: 16,  bias_hh: 16  -> 304 (RNN)
#   fc.weight: 1x16 = 16,  fc.bias: 1                                 ->  17 (Linear)
#   Total: ~321 parameters  (tiny! easy to train even on CPU)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model architecture:\n{model}\n")
print(f"Total trainable parameters: {total_params}")
print("  (Compare: the CNN in Ch 20 has 3.2 million parameters!)\n")

# ──────────────────────────────────────────────────────────
# 5.  TRAIN
# ──────────────────────────────────────────────────────────

# ---- nn.MSELoss() -- Mean Squared Error ----
# Computes:  loss = mean( (prediction - target)^2 )
# This is the standard loss for regression tasks (predicting a continuous value).
# For classification you would use nn.CrossEntropyLoss() instead.
# The lower the MSE, the closer our predictions are to the true sine values.
criterion = nn.MSELoss()

# ---- torch.optim.Adam(params, lr) ----
# Adam (Adaptive Moment Estimation) is the most popular optimiser in deep learning.
# It combines two ideas:
#   1. Momentum -- keeps a running average of past gradients (avoids oscillation)
#   2. Adaptive learning rate -- scales the step size per-parameter
#
# Parameters:
#   params : model.parameters() -- tells Adam which tensors to update
#   lr     : learning rate (step size).  Default is 0.001.
#       - Too large (e.g. 0.1)  -> training is unstable, loss may diverge
#       - Too small (e.g. 1e-6) -> training is very slow
#       - 0.01 works well for this small model; for larger models try 0.001
#
# Other optimisers you might see:
#   torch.optim.SGD   -- basic stochastic gradient descent (with optional momentum)
#   torch.optim.RMSprop -- adaptive LR only (no momentum averaging)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 50       # number of full passes through the training data
train_losses = []
val_losses   = []

for epoch in range(1, EPOCHS + 1):

    # ---- TRAINING STEP ----
    # model.train() -- switches the model to training mode.
    #   This matters for layers like Dropout and BatchNorm, which behave
    #   differently during training vs. inference.  Our simple RNN has
    #   neither, but it is good practice to always call it.
    model.train()

    # Forward pass: feed all training sequences through the model at once.
    # Because we set batch_first=True, X_train shape (N, 10, 1) is correct.
    pred = model(X_train)           # shape: (N, 1)
    loss = criterion(pred, Y_train) # scalar: mean squared error

    # ---- The 3-step backpropagation ritual ----
    # 1) optimizer.zero_grad()
    #      Clears old gradients from the previous step.  PyTorch ACCUMULATES
    #      gradients by default (useful for gradient accumulation tricks),
    #      so if you forget this, gradients keep growing and training breaks.
    optimizer.zero_grad()

    # 2) loss.backward()
    #      Computes d(loss)/d(param) for every trainable parameter using
    #      automatic differentiation ("autograd").  For an RNN this includes
    #      Back-Propagation Through Time (BPTT) -- gradients flow backward
    #      through every time step of the sequence.
    loss.backward()

    # 3) optimizer.step()
    #      Updates each parameter:  param = param - lr * grad
    #      (Adam actually uses a more sophisticated update rule.)
    optimizer.step()

    # ---- VALIDATION STEP ----
    # model.eval() -- switches to evaluation mode (affects Dropout etc.).
    model.eval()

    # torch.no_grad() -- disables gradient computation inside this block.
    #   Why?  During validation we only want to measure performance, not train.
    #   Skipping gradient tracking saves memory and speeds things up.
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, Y_val)

    # .item() converts a 1-element tensor to a plain Python float.
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train loss: {loss.item():.5f}  "
              f"Val loss: {val_loss.item():.5f}")

# ──────────────────────────────────────────────────────────
# 6.  VISUALISE RESULTS
# ──────────────────────────────────────────────────────────

# 6a. Loss curves
plt.figure(figsize=(8, 3))
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# 6b. Predictions vs ground truth on validation set
model.eval()
with torch.no_grad():
    # .squeeze() removes all dimensions of size 1.
    #   model(X_val) returns shape (N, 1); squeeze makes it (N,)
    #   so it plays nicely with matplotlib (which wants a 1-D array).
    # .numpy() converts the PyTorch tensor to a NumPy array for plotting.
    #   This only works on CPU tensors with no gradient tracking
    #   (which is why we are inside torch.no_grad()).
    preds = model(X_val).squeeze().numpy()
    truth = Y_val.squeeze().numpy()

plt.figure(figsize=(10, 3))
plt.plot(truth, "r-",  label="Actual (sin)")
plt.plot(preds, "g--", label="RNN prediction")
plt.xlabel("Validation sample index")
plt.ylabel("Value")
plt.title("RNN Predictions vs Actual Sine Values")
plt.legend()
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────
# 7.  KEY TAKEAWAYS
# ──────────────────────────────────────────────────────────
#
# a) WHY RNN and not a regular Dense network?
#    A Dense (fully connected) layer treats each input independently.
#    An RNN processes the sequence step by step, carrying a hidden state
#    h that accumulates context.  This makes it naturally suited for
#    ordered/temporal data (time series, text, audio, etc.).
#
# b) VANISHING / EXPLODING GRADIENTS
#    During BPTT the gradient is multiplied by W_hh at each time step.
#    Over many steps this product can shrink to ~0 ("vanishing") or
#    blow up ("exploding").  That is why plain RNNs struggle with long
#    sequences.  Solutions:
#      - Gradient clipping  (caps the gradient magnitude)
#      - LSTM / GRU layers  (have gating mechanisms designed to let
#        gradients flow over hundreds of steps).  In PyTorch simply
#        replace nn.RNN with nn.LSTM or nn.GRU -- same API!
#
# c) RNN vs LSTM vs GRU  (quick comparison)
#    ┌─────────┬──────────────┬────────────────────────────────────┐
#    | Layer   | Parameters   | When to use                        |
#    ├─────────┼──────────────┼────────────────────────────────────┤
#    | nn.RNN  | fewest       | short sequences, simple patterns   |
#    | nn.GRU  | medium       | good default; fewer params than LSTM|
#    | nn.LSTM | most         | long sequences, complex dependencies|
#    └─────────┴──────────────┴────────────────────────────────────┘
#
# ──────────────────────────────────────────────────────────
# 8.  EXERCISES FOR STUDENTS
# ──────────────────────────────────────────────────────────
"""
Try these modifications and observe how the results change:

1. WINDOW SIZE
   Change WINDOW from 10 to 3 or 30.
   -> Does a longer history help?  Is there a point of diminishing returns?

2. HIDDEN SIZE
   Change hidden_size from 16 to 4 or 64.
   -> How does capacity affect learning speed and accuracy?

3. NOISE
   Add noise to the sine wave:
       raw = np.sin(...) + 0.1 * np.random.randn(300)
   -> Can the RNN still learn the underlying pattern?

4. DIFFERENT WAVE
   Replace sin with a square wave or sawtooth (from scipy.signal) --
   does the RNN struggle with sharp transitions?

5. SWITCH TO LSTM
   Change nn.RNN to nn.LSTM in the model.  The forward() method stays
   almost the same (LSTM returns (output, (h_n, c_n)) instead of
   (output, h_n)).  Compare training speed and final loss.

6. MULTI-STEP PREDICTION (advanced)
   Instead of predicting only the next value, modify the model
   to predict the next 5 values at once (change output_size to 5
   and adjust the targets accordingly).
"""
