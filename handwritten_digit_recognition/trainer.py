import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import os

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(nn.Linear(input_size, hidden_size),
                                                  nn.Sigmoid(),
                                                  nn.Linear(hidden_size, output_size),
                                                  nn.Sigmoid(),)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(os.path.join(script_dir, 'training_data'), download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    net = NeuralNetwork(input_size=784,
                        hidden_size=100,
                        output_size=10,).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)

    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        running_batch = 1000
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images = data[0].float().to(device)
            labels = nn.functional.one_hot(data[1], num_classes=10).float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % running_batch == 0:
                print(f'[epoch {epoch + 1}, batch #{i + 1:4d}] loss: {running_loss / running_batch:.10f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = os.path.join(script_dir, 'mnist_net.pth')
    torch.save(net.state_dict(), PATH)
    print('saved trained nueral net!')
