"""
Small project to do image recognition on the CIFAR10 dataset
"""
import numpy as np
import torch
from torchsummary import summary
from torchvision import datasets, transforms

EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_WORKERS = 8


class NetworkStructure(torch.nn.Module):
    """
    Class module to describe the neural network structure
    """

    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(3, 16, kernel_size=5, padding=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module(
            "conv_2", torch.nn.Conv2d(16, 32, kernel_size=5, padding=2)
        )
        self.conv.add_module("relu_2", torch.nn.ReLU())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module(
            "conv_3", torch.nn.Conv2d(32, 32, kernel_size=5, padding=2)
        )
        self.conv.add_module("relu_3", torch.nn.ReLU())
        self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("flatten", torch.nn.Flatten())
        self.conv.add_module("dropout_1", torch.nn.Dropout(0.3))
        self.conv.add_module("fc_1", torch.nn.Linear(512, 10))
        self.conv.add_module("softmax", torch.nn.Softmax(dim=1))

    def forward(self, indata):
        """
        Forward pass of the network
        """

        return self.conv(indata)


def load_data():
    """
    Method to load and transform CIFAR10 dataset before training
    """

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])  # type: ignore
    return train_data, val_data


def network_step(data, optimizer, net, loss_function, train=True):
    """
    Do one step of the network (train or validation)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    if train:
        optimizer.zero_grad()
    outputs = net(inputs)
    loss = loss_function(outputs, labels)
    if train:
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the weights of the network
    _, max_indices = np.max(np.int32(outputs), 1)
    accuracy = (max_indices == labels).sum(dtype=np.int32) / max_indices.size(0)
    return loss, accuracy


def calculate_test_accuracy(net):
    """
    Module to calculate a trained networks test accuracy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    test_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)

        _, max_indices = np.max(np.int32(outputs), 1)
        correct += (max_indices == labels).sum(dtype=np.int32)
        total += max_indices.size(0)

    print(f"Test accuracy: {correct/total}")


def main():
    """
    Main method; loads data, trains network and evaluate it vs a test set
    """

    loss_function = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preparing dataset
    train_data, val_data = load_data()
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # Preparing network
    net = NetworkStructure()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    summary(net, (3, 32, 32))

    losses = np.zeros((2, EPOCHS))

    for epoch in range(EPOCHS):
        loss_sum = 0.0
        loss = 0.0
        accuracy = 0.0

        net.train()
        for i, data in enumerate(train_loader):
            network_loss, nbr_correct = network_step(
                data, optimizer, net, loss_function
            )

            # Collect losses and accuracy
            loss_sum += network_loss
            loss += network_loss
            accuracy += nbr_correct

            if i % 20 == 19:
                print(f"Epoch: {epoch+1}, Iteration: {i+1}, Loss: {loss_sum / 20}")
                loss_sum = 0.0
        losses[0, epoch] = loss / len(train_loader)

        # Validation loss
        loss = 0.0
        accuracy = 0.0
        net.eval()
        for data in val_loader:
            network_loss, nbr_correct = network_step(
                data, optimizer, net, loss_function
            )
            loss += network_loss
            accuracy += nbr_correct
        print(f"Validation loss: {loss / len(val_loader)}")
        print(f"Validation acc: {accuracy / len(val_loader)}")
        losses[1, epoch] = loss / len(val_loader)
    # Plot results


if __name__ == "__main__":
    main()
