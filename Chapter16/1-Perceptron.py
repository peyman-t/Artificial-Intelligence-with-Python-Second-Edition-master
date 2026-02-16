import neurolab as nl
import numpy as np

# 1. Prepare Data (Input: 2 dimensions, Target: 1 dimension)
# Example: AND logic gate pattern
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_data = np.array([[0], [0], [0], [1]])

# 2. Define Perceptron
# Slide 3: perceptron = nl.net.newp([dim1, dim2], num_output) [cite: 24]
# We define the min/max range for the 2 inputs
net = nl.net.newp([[0, 1], [0, 1]], 1)

# 3. Train the Network
# Slide 3: Train reduces error over epochs [cite: 25]
error_progress = net.train(input_data, target_data, epochs=100, show=20, lr=0.1)

# 4. Test
print("Testing [1, 1]:", net.sim([[1, 1]])) # Should be close to 1
print("Testing [0, 0]:", net.sim([[0, 0]])) # Should be close to 0
print("Testing [0, 1]:", net.sim([[0, 1]])) # Should be close to 0
print("Testing [1, 0]:", net.sim([[1, 0]])) # Should be close to 0