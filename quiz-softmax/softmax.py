#!/usr/bin/env python3

"""Softmax."""

import math
import numpy as np
import matplotlib.pyplot as plt

scores = [3.0, 1.0, 0.2]

# scores = [[1, 2, 3, 6],
#           [2, 4, 5, 6],
#           [3, 8, 7, 6]]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    # Original solution—needlessly verbose (still learning numpy)
    # return pow(math.e, np.array(x)) / pow(math.e, np.array(x)).sum(0)

    # Instructor's solution
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
