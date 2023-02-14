import numpy as np
from sklearn.utils import shuffle

def symmetry(d):
    image = d.reshape(16, 16)
    left_half = image[:,:8]
    right_half = image[:,-8:]
    flipped_right_half = np.fliplr(right_half)
    symmetry = 1-np.square((left_half-flipped_right_half)/2)
    
    return np.mean(symmetry)
    

def intensity(d):
    return np.mean(d)

def gradient_descent(X, y, step_size, max_iter, batch_size, reg, threshold):
    cur = 0 # current index
    loss_list = [np.inf] # loss values for each iteration
    
    # initialize weights
    w = np.zeros(X.shape[1], dtype=float)
    data = np.vstack((X.T,y)).T # attach y as last column vector to X
    def gradient(start, end):
        
        def theta(s):
            return 1/(1+np.exp(-s))
        
        def func(v):
            y_n = v[-1] # define last element as the label
            x_n = v[:-1] # define rest of the elements as data
            return -y_n*x_n*theta(-y_n*w.T@x_n)
        
        grad_per_datapoint = np.apply_along_axis(func, 1, data[start:end])
        g = np.mean(grad_per_datapoint, axis=0)
        return g + 2*reg*w
    
    def loss():
        def func(v):
            y_n = v[-1] # define last element as the label
            x_n = v[:-1] # define rest of the elements as data
            return np.log(1+np.exp(-y_n*w.T@x_n))
        
        loss_per_datapoint = np.apply_along_axis(func, 1, data)
        E_w = np.mean(loss_per_datapoint)
        return E_w + reg*np.linalg.norm(w)**2
    
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
            X, y, data = shuffle(X, y, data)
        else:
            cur = nex
        
        loss_list.append(loss())
        # print(f'loss after {n}th decent: {loss_list[n]}')
        if abs(loss_list[n-1] - loss_list[n]) < threshold:
            print("Converged after " + str(n) + " gradient computations")
            return w, loss_list
        
    print(f'Failed to converge even after {max_iter} gradient computations')
    return w, loss_list