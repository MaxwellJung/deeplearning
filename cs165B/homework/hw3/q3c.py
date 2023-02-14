import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.load('train_data.npy')
label = np.load('train_labels.npy')

# show images
def main():
    symmetry_data = np.apply_along_axis(symmetry, 1, data)
    intensity_data = np.apply_along_axis(intensity, 1, data)
    
    one = label == 1
    five = label == -1
    
    plt.plot(symmetry_data[five], intensity_data[five], 'rx', label='5')
    plt.plot(symmetry_data[one], intensity_data[one], 'bo', label='1')
    plt.legend()
    plt.xlabel("symmetry")
    plt.ylabel("intensity")
    plt.show()
    
def symmetry(d):
    image = d.reshape(16, 16)
    left_half = image[:,:8]
    right_half = image[:,-8:]
    flipped_right_half = np.fliplr(right_half)
    symmetry = 1-np.square((left_half-flipped_right_half)/2)
    
    return np.mean(symmetry)
    

def intensity(d):
    return np.mean(d)

if __name__ == '__main__':
    main()