import numpy as np
from numpy import linalg as LA

a = np.array([[2, 1, 3],
              [1, 1, 2],
              [3, 2, 5]])

eigenvectors = LA.eig(a)[1]

print(eigenvectors)

for i in range(3):
    print(a @ eigenvectors[:,i])
    print(a @ eigenvectors[:,i]/eigenvectors[:,i])