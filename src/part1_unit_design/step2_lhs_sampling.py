import numpy as np
from pyDOE import lhs
import pandas as pd

num = 3000

# Create space with x, y, and z dimensions
x = np.arange(2, 14.1, 0.1)
y = np.arange(2, 20.1, 0.1)
z = np.array([1.24, 2.25, 2.53, 2.72])

# Flatten the space into a 2D matrix
space = np.zeros((len(x) * len(y) * len(z), 4))
space[:, 0] = np.arange(1, len(x) * len(y) * len(z) + 1)
space[:, 1] = np.repeat(x, len(y) * len(z))
space[:, 2] = np.tile(np.repeat(y, len(z)), len(x))
space[:, 3] = np.tile(z, len(x) * len(y))

# Generate samples using Latin Hypercube Sampling (LHS)
X = lhs(4, samples=num, criterion='maximin')
Y = np.round((len(x) * len(y) * len(z) - 1) * X).astype(int)
samples = space[Y, :]

# Save space and samples to CSV files
np.savetxt("../data/dataset/space.csv", space, delimiter=",")
np.savetxt("../data/dataset/samples.csv", samples, delimiter=",")

# Read and display the samples matrix
samples = pd.read_csv("../data/dataset/samples.csv", header=None)
print(samples)
