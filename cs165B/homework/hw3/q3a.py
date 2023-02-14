import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.load('train_data.npy')
label = np.load('train_labels.npy')

# show images
while(True):
    i=np.random.randint(1561) # pick from 0 to 1560
    plt.figure()
    plt.imshow(data[i].reshape(16, 16), cmap=plt.cm.binary)
    plt.show()
    print(label[i])