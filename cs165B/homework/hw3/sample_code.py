import numpy as np
import matplotlib.pyplot as plt



# load data
data = np.load('train_data.npy')
label = np.load('train_labels.npy')

# show images
i=1200
plt.figure()
plt.imshow(data[i].reshape(16, 16), cmap=plt.cm.binary)
plt.show()
print(label[i])

# flip image
plt.figure()
plt.imshow(data[i].reshape(16, 16), cmap=plt.cm.binary)
plt.show()
print(label[i])
plt.imshow(np.fliplr(data[i].reshape(16, 16)), cmap=plt.cm.binary)
plt.show()

# plot points
plt.figure()
sample_data1 = np.random.randn(10, 2)
sample_data2 = np.random.randn(10, 2)
plt.plot(sample_data1[:, 0], sample_data1[:, 1], 'rx', label='5')
plt.plot(sample_data2[:, 0], sample_data2[:, 1], 'bo', label='1')
plt.legend()
plt.show()