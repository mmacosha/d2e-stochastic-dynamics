import os

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm.auto import trange

from models.model import MNISTEnergy


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the MNIST dataset
    data_path = "~/mnist/data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True,transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )

    # Initialize the network, loss function, and optimizer
    net = MNISTEnergy(as_cls=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training the network
    for epoch in trange(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}]', 
                      f' loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    os.makedirs('~/mnist/checkpoints', exist_ok=True)
    torch.save(net.state_dict(),'~/mnist/checkpoints/mnist_classifier.pth')

if __name__ == '__main__':
    main()