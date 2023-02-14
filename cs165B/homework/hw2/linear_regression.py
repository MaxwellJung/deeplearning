import numpy as np
from sklearn.utils import shuffle

def compare(u,v):
    return np.linalg.norm(u-v)

def generate_data(sample_size: int, feature_size: int):
    X = np.random.randn(sample_size, feature_size)
    X = np.concatenate((X, np.ones((sample_size, 1))), axis=1)

    # Generate ground truth model.
    w_gt = np.random.randn(feature_size + 1, 1) * 10

    # Generate label.
    y = X @ w_gt

    # Add element-wise gaussian noise to each label.
    y += np.random.randn(sample_size, 1)

    return w_gt, X, y

def lsqr_closed_form(X,y):
    return np.linalg.inv(X.T@X)@X.T@y

def ridge_closed_form(X,y, reg):
    return np.linalg.inv(X.T@X+reg*np.identity(X.shape[1]))@X.T@y

def gradient_descent(X, y, step_size, max_iter, batch_size, reg):
    cur = 0 # current index
    loss_list = [np.inf] # loss values for each iteration
    
    # initialize weights
    w = np.zeros([X.shape[1], 1], dtype=float)
    
    def gradient(start, end):
        return X[start:end].T@X[start:end]@w - X[start:end].T@y[start:end] + reg*w
    
    def loss():
        return 0.5 * np.linalg.norm(X@w-y)**2 + 0.5 * reg * np.linalg.norm(w)**2
    
    for n in range(1, max_iter):
        nex = (cur + batch_size) if (cur + batch_size < X.shape[0]) else X.shape[0]

        # adjust weights: w = w - step_size * dir
        g = gradient(cur, nex)
        dir = -1 * g/np.linalg.norm(g)
        w += step_size/np.log(n+1) * dir
        
        
        if nex == X.shape[0]:
            # move back to beginning
            cur = 0
            
            # shuffle data
            X, y = shuffle(X, y)
        else:
            cur = nex
        
        loss_list.append(loss())
        # print(f'loss after {n}th decent: {loss_list[n]}')
        if abs(loss_list[n-1] - loss_list[n]) < 1e-4:
            print("Converged after " + str(n) + " gradient computations")
            return w, loss_list
        
    print(f'Failed to converge even after {max_iter} gradient computations')
    return w, loss_list