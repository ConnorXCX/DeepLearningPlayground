import numpy as np

x = np.random.random((10,))
print(x)

x = np.expand_dims(x, axis=0)
x = np.concatenate([x] * 32, axis=0)

print(x)
