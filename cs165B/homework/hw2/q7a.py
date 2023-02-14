from linear_regression import *
import matplotlib.pyplot as plt

def main():
    sample_count = 1000
    feature_count = 100
    w_gt, X, y = generate_data(sample_count, feature_count)
    n = 100
    X_test = X[:n] # get first n data points
    y_test = y[:n] # get first n data points
    X_training = X[n:] # get everything after nth data points
    y_training = y[n:] # get everything after nth data points

    reg = 1e-6
    
    x_axis = range(1, sample_count) # 50, 100, 150, ..., 950
    mse_list = []
    for i in x_axis:
        w = ridge_closed_form(X_training[:i], y_training[:i], reg)
        
        mse = np.linalg.norm(X_test@w-y_test)**2
        mse_list.append(mse)
    
    plt.plot(x_axis, mse_list, c='red')
    
    plt.legend(['training data size vs error'])
    plt.show()
    
    return

if __name__ == '__main__':
    main()