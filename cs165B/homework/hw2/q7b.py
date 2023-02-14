from linear_regression import *
import matplotlib.pyplot as plt

def main():
    sample_count = 1000
    max_feature_count = 500
    n = 100
    reg = 1e-6
    
    x_axis = range(1, max_feature_count) # 50, 100, 150, ..., 450
    mse_list = []
    for feature_count in x_axis:
        w_gt, X, y = generate_data(sample_count, feature_count)
        
        X_test = X[:n] # get first n data points
        y_test = y[:n] # get first n data points
        X_training = X[n:] # get everything after nth data points
        y_training = y[n:] # get everything after nth data points
        
        w = ridge_closed_form(X_training, y_training, reg)
        
        mse = np.linalg.norm(X_test@w-y_test)**2
        mse_list.append(mse)
    
    plt.plot(x_axis, mse_list, c='red')
    
    plt.legend(['training data dimension vs error'])
    plt.show()
    
    return

if __name__ == '__main__':
    main()