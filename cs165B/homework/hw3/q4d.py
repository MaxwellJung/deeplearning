import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import defaultdict
from logistic_regression import *

# load data
train_data = np.load('train_data.npy')
train_label = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_labels.npy')

# show images
def main():
    max_iter = 10000
    step_size = 1
    batch_size = 100
    reg = np.exp(-50)
    
    s = np.apply_along_axis(symmetry, 1, train_data)
    i = np.apply_along_axis(intensity, 1, train_data)
    
    X = np.stack((np.ones(len(train_data)), 
                  s, 
                  i, 
                  s**2,
                  s*i,
                  i**2,
                  s**3,
                  s**2*i,
                  s*i**2,
                  i**3)).T
    y = train_label
    X, y = shuffle(X, y)
    
    w, loss_list = gradient_descent(X, y, step_size, max_iter, batch_size, reg, 1e-5)
    print(w)
    
    plt.figure()
    plt.title('Training')
    graph(train_data, train_label, w)
    E_training = loss(w, X, y)
    
    st = np.apply_along_axis(symmetry, 1, test_data)
    it = np.apply_along_axis(intensity, 1, test_data)
    X_test = np.stack((np.ones(len(test_data)), 
                       st, 
                       it, 
                       st**2,
                       st*it,
                       it**2,
                       st**3,
                       st**2*it,
                       st*it**2,
                       it**3)).T
    y_test = test_label
    
    plt.figure()
    plt.title('Test')
    graph(test_data, test_label, w)
    E_test = loss(w, X_test, y_test)
    
    print(f'Training error: {E_training}')
    print(f'Test error: {E_test}')
    
    plt.show()
    
def graph(data, label, w):
    symmetry_data = np.apply_along_axis(symmetry, 1, data)
    intensity_data = np.apply_along_axis(intensity, 1, data)
    
    one = label == 1
    five = label == -1
    
    plt.plot(symmetry_data[five], intensity_data[five], 'rx', label='5')
    plt.plot(symmetry_data[one], intensity_data[one], 'bo', label='1')
    draw_line(w, 0.6, 1.01, color='green')
    plt.legend()
    plt.xlabel("symmetry")
    plt.ylabel("intensity")
    
def draw_line(normal, min, max, color):
    x = np.arange(min, max, 0.1)
    m = -normal[1]/normal[2]
    b = -normal[0]/normal[2]
    y = m*x+b
    
    plt.plot(x, y, c=color, label='w')
    
def loss(w, X, y):
    Xy = np.vstack((X.T,y)).T
    
    def func(v):
        y_n = v[-1] # define last element as the label
        x_n = v[:-1] # define rest of the elements as data
        return np.log(1+np.exp(-y_n*w.T@x_n))
    
    loss_per_datapoint = np.apply_along_axis(func, 1, Xy)
    return np.mean(loss_per_datapoint)
    
if __name__ == '__main__':
    main()