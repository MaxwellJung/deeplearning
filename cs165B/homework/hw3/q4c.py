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
    batch_size = 1561
    reg_list = [np.exp(i) for i in range(-100,0)]
    
    group_count = 3
    group_size = batch_size//group_count
    
    symmetry_data = np.apply_along_axis(symmetry, 1, train_data)
    intensity_data = np.apply_along_axis(intensity, 1, train_data)
    X = np.stack((np.ones(len(train_data)), symmetry_data, intensity_data)).T
    y = train_label
    X, y = shuffle(X, y)
    
    d = defaultdict(list)
    
    for g in range(group_count):
        for reg in reg_list:
            print(f'trying ln(lambda) = {np.log(reg)}')
            w, loss_list = gradient_descent(np.vstack([X[0: g*group_size], X[(g+1)*group_size:]]), 
                                            np.concatenate([y[0: g*group_size], y[(g+1)*group_size:]]), 
                                            1, 
                                            1000, 
                                            batch_size, 
                                            reg,
                                            1e-4)
            print(w)
            e = loss(w, 
                X[g*group_size:(g+1)*group_size], 
                y[g*group_size:(g+1)*group_size])
            
            d[reg].append(e)
            
    for reg in d:
        d[reg] = np.mean(d[reg])
        
    best_reg = min(d, key=d.get)
    print(f'best regularization factor: ln(lambda) = {np.log(best_reg)}')
    w, loss_list = gradient_descent(X, y, step_size, max_iter, batch_size, best_reg, 1e-5)
    print(w)
    
    plt.figure()
    plt.title('Training')
    graph(train_data, train_label, w)
    E_training = loss(w, X, y)
    
    X_test = np.stack((np.ones(len(test_data)), 
                  np.apply_along_axis(symmetry, 1, test_data), 
                  np.apply_along_axis(intensity, 1, test_data))).T
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