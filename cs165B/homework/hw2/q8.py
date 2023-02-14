import time
from linear_regression import *
import matplotlib.pyplot as plt

def main():
    sample_count = 1000
    max_feature_count = 2000
    n = 100
    reg = 1e-6
    
    x_axis = range(1, max_feature_count) # 50, 100, 150, ..., 450
    computation_time_list = []
    for feature_count in x_axis:
        w_gt, X, y = generate_data(sample_count, feature_count)
        
        X_test = X[:n] # get first n data points
        y_test = y[:n] # get first n data points
        X_training = X[n:] # get everything after nth data points
        y_training = y[n:] # get everything after nth data points
        
        start = time.time()
        w = ridge_closed_form(X_training, y_training, reg)
        end = time.time()
        computation_time = end-start
        print(f'completed training on {feature_count} dimension data in {computation_time} seconds')
        
        computation_time_list.append(computation_time)
    
    plt.plot(x_axis, computation_time_list, c='red')
    
    plt.legend(['training data dimension vs computation time'])
    plt.show()
    
    return

if __name__ == '__main__':
    main()