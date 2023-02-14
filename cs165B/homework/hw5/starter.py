import numpy as np
from cvxopt import matrix, solvers


X = np.array([[0, 0], [0, -1], [-2, 0]]) # 3 x 2
y = np.array([-1, -1, 1]).reshape(3, 1)  # 3 x 1

dim = X.shape[1]  # dimensionality
num = X.shape[0]  # sample size

Q = np.eye(dim+1)
Q[0, 0] = 0
p = np.zeros((dim+1, 1))
A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
c = np.ones((num, 1))

sol = solvers.qp(P=matrix(Q), q=matrix(p), G=matrix(-A), h=matrix(-c))
print(sol['x'])