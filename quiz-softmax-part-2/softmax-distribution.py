#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

scores = np.array([3.0, 1.0, 0.2])

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

y = np.vstack([softmax(scores), softmax(scores * 10)])

plt.scatter([scores, scores], y, s=(np.pi * 5**2), c='rrrbbb')
plt.show()
