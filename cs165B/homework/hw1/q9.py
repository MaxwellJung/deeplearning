import numpy as np
import matplotlib.pyplot as plt

def main():
    dimension = 2
    data_set_count = 20
    min_value = -50
    max_value = 50
    
    data_set = generate_data_set(dimension, data_set_count, min_value, max_value)
    f = data_set['f']
    x = data_set['x']
    y = data_set['y']
    
    positive = x[y > 0]
    negative = x[y < 0]
    
    g = 2*np.random.rand(dimension+1)-1
    
    t = 0
    while True:
        improved_g = improve(g, x, y)
        if np.array_equal(g, improved_g):
            break
        else:
            g = improved_g
        t += 1
        
    print(f'updated perception {t} times')
    
    plt.scatter(positive[:, 1], positive[:, 2], c='blue')
    plt.scatter(negative[:, 1], negative[:, 2], c='red')
    
    plot(f, min_value, max_value, color='black')
    plot(g, min_value, max_value, color='green')
    
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(f'Linear Classifier for {data_set_count} data points')
    plt.legend(['Y = 1', 'Y = -1', 'Target', 'Hypothesis'])
    plt.show()
    
def plot(normal, min, max, color):
    x = np.arange(min, max, 0.1)
    m = -normal[1]/normal[2]
    b = -normal[0]/normal[2]
    y = m*x+b
    
    plt.plot(x, y, c=color)

def improve(g, x, y):
    expected = np.sign(g @ x.T)
    actual = y
    misclassified = np.not_equal(expected, actual)
    
    if misclassified.any():
        missed_idx = np.argwhere(misclassified).T[0]
        i = np.random.randint(len(missed_idx))
        random_missed_index = missed_idx[i]
        
        x_star = x[random_missed_index]
        y_star = y[random_missed_index]
        
        return g + y_star*x_star
    else:
        return g

def generate_data_set(dimension, n, min_value, max_value):
    f = 2*np.random.rand(dimension+1)-1
    
    ones_vector = np.ones((n, 1))
    x = (max_value-min_value)*np.random.rand(n, dimension)-(max_value-min_value)/2
    x = np.hstack((ones_vector, x))
    y = np.sign(f @ x.T)
    
    return {'f': f, 'x': x, 'y': y}

if __name__ == '__main__':
    main()