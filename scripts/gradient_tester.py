import numpy as np

grad = np.gradient(np.array([[1, 2, 6], [5, 5, 5]], dtype=float), axis=0)

print(grad[0])