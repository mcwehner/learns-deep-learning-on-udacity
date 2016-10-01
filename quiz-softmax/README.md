## Quiz: Softmax

### Notes

#### Column vectors in `numpy`

My first solution was _almost_ correct, but I failed to specify the `axis` when using `sum()`. This worked for a single column vector of values, but not for a two-dimensional matrix of values.

```python
# Worked accidentally, despite not including `axis`.
scores = [3.0, 1.0, 0.2]

np.sum(scores)  # => ~4.2

# Didn't work: it sums everything, producing a single scalar result.
scores = [[1, 2, 3, 6],
          [2, 4, 5, 6],
          [3, 8, 7, 6]]

np.sum(scores)  # => 53

# Works: needed to specify axis 0 for column vectors.
np.sum(scores, axis=0)  # => [ 6, 14, 15, 18]
```


#### Personal ignorance in `numpy`

Because I'm only just beginning to purposefully familiarize myself with the wealth of functionality in `numpy`, my original implementation of `softmax` was needlessly verbose:

```python
import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return pow(math.e, np.array(x)) / pow(math.e, np.array(x)).sum(0)
```


### TODO

* Need to set aside some time to explore `numpy` on its own, in depth.

### Resources

* [Softmax: Neural Networks and Deep Learning, Chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)
* [`numpy` array methods](http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#array-methods)
* [`math` library documentation](https://docs.python.org/3/library/math.html)
* [Summation (Wikipedia)](https://en.wikipedia.org/wiki/Summation)
