import neurolab as nl
import numpy as np

# 1. Create Data Clusters
# Slide 8: Input data (black dots) scattered in groups [cite: 141]
# Group A (around 0,0) and Group B (around 5,5)
data = np.array([[0.1, 0.1], [0.2, 0.2], [5.1, 5.1], [5.2, 5.2]])
# Labels: Class 0 for Group A, Class 1 for Group B
labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

# 2. Create LVQ Network
# Slide 7: nn = nl.net.newlvq(minmax, neurons, weights) [cite: 138]
# We need MORE competitive neurons than classes (cn0 > num_classes)
# 4 neurons for 2 classes (2 per class), with equal class proportions
net = nl.net.newlvq(nl.tool.minmax(data), 4, [0.5, 0.5])

# 3. Train
error = net.train(data, labels, epochs=500, goal=0.01)

# 4. Interpretability Check
# Slide 8: LVQ offers high interpretability [cite: 142]
print("Class for [0.1, 0.1]:", net.sim([[0.3, 0.3]]))
print("Class for [5.1, 5.1]:", net.sim([[5.1, 5.1]]))