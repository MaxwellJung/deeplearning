from linear_regression import *
import matplotlib.pyplot as plt

def main():
    sample_count = 5000
    feature_count = 1000
    max_iter = 10000
    step_size = 1
    reg = 1e-6
    w_gt, X, y = generate_data(sample_count, feature_count)
    
    ridge = ridge_closed_form(X, y, reg)
    sgd_1, sgd_1_errors = gradient_descent(X, y, step_size, max_iter, 1, reg)
    sgd_10, sgd_10_errors = gradient_descent(X, y, step_size, max_iter, 10, reg)
    sgd_100, sgd_100_errors = gradient_descent(X, y, step_size, max_iter, 100, reg)
    
    plt.plot(range(len(sgd_1_errors)), sgd_1_errors, c='red')
    plt.plot(range(len(sgd_10_errors)), sgd_10_errors, c='orange')
    plt.plot(range(len(sgd_100_errors)), sgd_100_errors, c='blue')
    
    plt.legend(['batch size = 1', 'batch size = 10', 'batch size = 100'])
    plt.show()
    
    return

if __name__ == '__main__':
    main()