import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn
import os
from trainer import NeuralNetwork

script_dir = os.path.dirname(os.path.realpath(__file__))

transform = transforms.Compose([transforms.ToTensor()])

testset = datasets.MNIST(os.path.join(script_dir, 'test_data'), download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
    
net = NeuralNetwork(input_size=784,
                    hidden_size=100,
                    output_size=10,).to(device)

PATH = os.path.join(script_dir, 'mnist_net.pth')
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
wrong_images = []
wrong_guesses = []
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        guessed_probabilities, guessed_digits = torch.max(outputs.data, dim=1)
        
        misclassified = guessed_digits != labels
        wrong_images.append(images[misclassified])
        wrong_guesses.append(guessed_digits[misclassified])
        
        total += labels.size(0)
        correct += (guessed_digits == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct/total:.2f} %')

wrong_images = torch.cat(wrong_images)
wrong_guesses = torch.cat(wrong_guesses)

num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(wrong_images[index][0].cpu(), cmap='gray_r')
    
print(f'Showing incorrectly guessed images')
# print(f'guessed: {wrong_guesses}')
plt.show()