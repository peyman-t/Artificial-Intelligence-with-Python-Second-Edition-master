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
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# 0.  SEED for reproducibility
# ──────────────────────────────────────────────────────────
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

raw = np.sin(np.linspace(0, 20 * np.pi, 300)).astype(np.float32)

# Quick look at the data
plt.figure(figsize=(10, 2))
plt.title("Raw sine-wave data (300 points)")
plt.plot(raw)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────
# 2.  CREATE SEQUENCES  (sliding window)
# ──────────────────────────────────────────────────────────
WINDOW = 10          # how many past values the RNN sees at each step

def make_sequences(data, window):
    """Slide a window over `data`; return (inputs, targets) tensors."""
    xs, ys = [], []
    for i in range(len(data) - window):
        xs.append(data[i : i + window])
        ys.append(data[i + window])          # next value after the window
    # shapes:  X → (N, window, 1)   Y → (N, 1)
    X = torch.tensor(np.array(xs)).unsqueeze(-1)   # add feature dim
    Y = torch.tensor(np.array(ys)).unsqueeze(-1)
    return X, Y

X, Y = make_sequences(raw, WINDOW)

# Train / validation split (first 80 % train, rest validation)
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val,   Y_val   = X[split:], Y[split:]

print(f"Training examples : {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Each input shape   : {tuple(X_train[0].shape)}  (window × features)")

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
hidden_size = 4

# Random weights (pretend they are learned)
W_ih = torch.randn(hidden_size, input_size)
W_hh = torch.randn(hidden_size, hidden_size)
bias = torch.zeros(hidden_size)

h = torch.zeros(hidden_size)            # initial hidden state
demo_seq = X_train[0]                   # shape (WINDOW, 1)

for t in range(3):                      # just show first 3 steps
    x_t = demo_seq[t]                   # shape (1,)
    h = torch.tanh(W_ih @ x_t + W_hh @ h + bias)
    print(f"  t={t}  x_t={x_t.item():.3f}  ->  h_t = {h.numpy().round(3)}")

print("  ... (h keeps updating at every time step)\n")

# ──────────────────────────────────────────────────────────
# 4.  DEFINE THE MODEL
# ──────────────────────────────────────────────────────────
#   Architecture (intentionally tiny):
#       Input (1 feature per step)
#         → RNN layer  (hidden_size = 16)
#         → Linear     (16 → 1)        ← predicts next sine value
#
#   Total parameters: ~300  (vs. 3.2 M in the CNN from Ch 20!)

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        rnn_out, h_n = self.rnn(x)
        # rnn_out shape: (batch, seq_len, hidden_size)
        # We only need the output at the LAST time step
        last_hidden = rnn_out[:, -1, :]       # (batch, hidden_size)
        prediction  = self.fc(last_hidden)    # (batch, output_size)
        return prediction

model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model architecture:\n{model}\n")
print(f"Total trainable parameters: {total_params}\n")

# ──────────────────────────────────────────────────────────
# 5.  TRAIN
# ──────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 50
train_losses = []
val_losses   = []

for epoch in range(1, EPOCHS + 1):
    # --- training step ---
    model.train()
    pred = model(X_train)
    loss = criterion(pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- validation step (no gradient) ---
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, Y_val)

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
# 7.  EXERCISES FOR STUDENTS
# ──────────────────────────────────────────────────────────
"""
Try these modifications and observe how the results change:

1. WINDOW SIZE
   Change WINDOW from 10 to 3 or 30.
   → Does a longer history help?  Is there a point of diminishing returns?

2. HIDDEN SIZE
   Change hidden_size from 16 to 4 or 64.
   → How does capacity affect learning speed and accuracy?

3. NOISE
   Add noise to the sine wave:
       raw = np.sin(...) + 0.1 * np.random.randn(300)
   → Can the RNN still learn the underlying pattern?

4. DIFFERENT WAVE
   Replace sin with a square wave or sawtooth — does the RNN
   struggle with sharp transitions?

5. MULTI-STEP PREDICTION (advanced)
   Instead of predicting only the next value, modify the model
   to predict the next 5 values at once (change output_size to 5
   and adjust the targets accordingly).
"""
