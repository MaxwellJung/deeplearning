import numpy as np
from numpy.linalg import matrix_rank

a1 = np.array([1,2,4])
b1 = np.array([3,-5,1])

a2 = np.array([[1,4,-3],
               [2,-1,3]])
b2 = np.array([[-2,0,5],
               [0,-1,4]])

print(f'inner product of {a1} and {b1} is {a1 @ b1}')
print(f'inner product of {a2.T} and {b2} is {a2.T @ b2} with rank {matrix_rank(a2.T @ b2)}')
print(f'inner product of {b2} and {a2.T} is {b2 @ a2.T} with rank {matrix_rank(b2 @ a2.T)}')