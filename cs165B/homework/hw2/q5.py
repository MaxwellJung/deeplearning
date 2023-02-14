from linear_regression import *
import matplotlib.pyplot as plt

def main():
    sample_count = 1000
    feature_count = 50
    batch_size = sample_count//5
    max_iter = 10000
    step_size = 1
    reg = 1e-6
    w_gt, X, y = generate_data(sample_count, feature_count)
    
    lsqr = lsqr_closed_form(X,y)
    lsqr_gd, lsqr_gd_errors = gradient_descent(X, y, step_size, max_iter, sample_count, 0)
    lsqr_sgd, lsqr_sgd_errors = gradient_descent(X, y, step_size, max_iter, batch_size, 0)
    
    ridge = ridge_closed_form(X, y, reg)
    ridge_gd, ridge_gd_errors = gradient_descent(X, y, step_size, max_iter, sample_count, reg)    
    ridge_sgd, ridge_sgd_errors = gradient_descent(X, y, step_size, max_iter, batch_size, reg)
    
    print(f'lsqr regression results:')
    print(f'GD vs SGD: {compare(lsqr_gd, lsqr_sgd)}')
    print(f'GD vs closed-form: {compare(lsqr_gd, lsqr)}')
    print(f'SGD vs closed-form: {compare(lsqr_sgd, lsqr)}')
    print(f'w_gt vs closed-form: {compare(w_gt, lsqr)}')
    print('-'*100)
    print(f'ridge regression results:')
    print(f'GD vs SGD: {compare(ridge_gd, ridge_sgd)}')
    print(f'GD vs closed-form: {compare(ridge_gd, ridge)}')
    print(f'SGD vs closed-form: {compare(ridge_sgd, ridge)}')
    print(f'w_gt vs closed-form: {compare(w_gt, ridge)}')
    
    plt.plot(range(len(lsqr_gd_errors)), lsqr_gd_errors, c='red')
    plt.plot(range(len(lsqr_sgd_errors)), lsqr_sgd_errors, c='orange')
    plt.plot(range(len(ridge_gd_errors)), ridge_gd_errors, c='blue')
    plt.plot(range(len(ridge_sgd_errors)), ridge_sgd_errors, c='skyblue')
    
    plt.legend(['lsqr_gd', 'lsqr_sgd', 'ridge_sg', 'ridge_sgd'])
    plt.show()
    
    return

if __name__ == '__main__':
    main()