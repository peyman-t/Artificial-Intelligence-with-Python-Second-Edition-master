import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Non-Linear Data (Parabola)
# Slide 5: Real-world data is rarely linear (e.g., y = 3x^2 + 5) [cite: 81]
x = np.linspace(-15, 15, 50).reshape(50, 1)
y = 3 * (x ** 2) + 5
# Normalize data for better training (crucial for Neural Networks)
y = y / np.max(y) 

# 2. Create Feedforward Network (Multi-Layer)
# Slide 5: nn = nl.net.newff([min, max], [10, 6, 1]) [cite: 105]
# Structure: Input -> 10 Hidden -> 6 Hidden -> 1 Output
net = nl.net.newff([[-15, 15]], [10, 6, 1])

# 3. Set Training Algorithm
# Slide 5: Gradient Descent [cite: 105]
net.trainf = nl.train.train_gd

# 4. Train
# Slide 6: Visualizing Gradient Descent hugging the curve [cite: 107, 117]
error = net.train(x, y, epochs=4000, goal=0.01, show=100)

# 5. Visualize
prediction = net.sim(x)
plt.plot(x, y, '-', label='Actual')
plt.plot(x, prediction, 'r.', label='Predicted')
plt.legend()
plt.show()