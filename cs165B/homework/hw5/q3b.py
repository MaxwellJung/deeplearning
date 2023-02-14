import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import timeit

max_dimension = 500
max_samples = 1000
lower_bound = -1000
upper_bound = 1000

def generate_data_set(dimension, samples, lower_bound, upper_bound):
    w = 2*(np.random.rand(dimension)-0.5)
    b = 2*(np.random.rand(1)-0.5)
    
    x = (upper_bound-lower_bound)*np.random.rand(samples, dimension)-(upper_bound-lower_bound)/2
    y = np.sign(w @ x.T + b).reshape(-1, 1)
    
    return {'w': w, 'b': b, 'x': x, 'y': y}

def generate_pqgh(X, y):
    dim = X.shape[1]  # dimensionality
    num = X.shape[0]  # sample size

    Q = np.identity(dim+1)
    Q[0, 0] = 0
    p = np.zeros((dim+1, 1))
    A = np.array(np.concatenate((y, y * X), axis=1)).astype(float)
    c = np.ones((num, 1))
    
    return matrix(Q), matrix(p), matrix(-A), matrix(-c)

        
    u = np.array(sol['x']).flatten()
    return u

def plot(normal, lower_bound, upper_bound, color):
    x = np.arange(lower_bound, upper_bound, 0.1)
    m = -normal[1]/normal[2]
    b = -normal[0]/normal[2]
    y = m*x+b
    
    plt.plot(x, y, c=color)
    
if __name__ == '__main__':
    dataset = generate_data_set(200, max_samples, lower_bound, upper_bound)

    sample_count = range(1, max_samples)
    execution_times_s = []
    
    for i in sample_count:
        X = dataset['x'][:i]
        y = dataset['y'][:i]
        args = generate_pqgh(X, y)
        solvers.options['show_progress'] = False
        execution_time = timeit.timeit("solvers.qp(*args)", globals=globals(), number=1)
        print(f'samples: {i} time: {execution_time}')
        execution_times_s.append(execution_time)
    
        
    dimension_count = range(1, max_dimension)
    execution_times_d = []
    
    for i in dimension_count:
        dataset = generate_data_set(i, max_samples, lower_bound, upper_bound)
        X = dataset['x']
        y = dataset['y']
        args = generate_pqgh(X, y)
        solvers.options['show_progress'] = False
        execution_time = timeit.timeit("solvers.qp(*args)", globals=globals(), number=1)
        print(f'dimension: {i} time: {execution_time}')
        execution_times_d.append(execution_time)
    
    plt.figure()
    plt.title('sample size vs QP execution time')
    plt.plot(sample_count, execution_times_s, c='red')
    
    plt.figure()
    plt.title('dimension size vs QP execution time')
    plt.plot(dimension_count, execution_times_d, c='red')
    
    plt.show()