import time

import numpy as np


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


def main():
    # NumPy implementation
    x = np.random.random((20, 100))
    y = np.random.random((20, 100))

    t0 = time.time()
    for _ in range(10000):
        z = x + y
        np.maximum(z, 0.)
    print('Took: {0:.2f} s'.format(time.time() - t0))

    # Naive implementation
    t0 = time.time()
    for _ in range(10000):
        z = naive_add(x, y)
        naive_relu(z)
    print('Took: {0:.2f} s'.format(time.time() - t0))


main()
