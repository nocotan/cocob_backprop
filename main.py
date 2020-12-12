import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST

from cocob_backprop import COCOBBackprop


# set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

datasets = ["mnist", "cifar10"]
optims = ["adam", "amsgrad", "rmsprop", "cocob_backprop"]


class MLP(nn.Module):
    def __init__(self, input_features, n_classes):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


def main():
    # set argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=datasets, default="mnist")
    parser.add_argument("--optimizer", choices=optims, default="adam")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()

    # set manual seed
    torch.manual_seed(args.seed)

    root = "./data"
    if args.dataset == "mnist":
        input_features = 28*28
        n_classes = 10
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = MNIST(root, train=True, download=True, transform=transform)
        test_set = MNIST(root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    net = MLP(input_features=input_features, n_classes=n_classes)
    net = net.to(device)

    if args.optimizer == "cocob_backprop":
        optimizer = COCOBBackprop(net.parameters())
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    elif args.optimizer == "amsgrad":
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0005)
    else:
        raise NotImplementedError

    train_losses = []
    test_losses = []
    for epoch in range(args.n_epochs):
        train_loss = train(net, optimizer, train_loader)
        test_loss = test(net, test_loader)
        print("Epoch={}, train loss={:.4f}, test loss={:.4f}".format(
            epoch, train_loss, test_loss))

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    save_csv(args, train_losses, test_losses)


def train(net, optimizer, train_loader):
    net.train()
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = F.nll_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    return epoch_loss / len(train_loader)


def test(net, test_loader):
    net.eval()
    test_loss = 0.0
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = F.nll_loss(outputs, targets)
        test_loss += loss.detach().item()

    return test_loss / len(test_loader)


def save_csv(args, train_losses, test_losses):
    fname = "{}_{}_{}.csv".format(
        args.optimizer,
        args.dataset,
        args.n_epochs)

    df = pd.DataFrame(
        {"train_losses": train_losses, "test_losses": test_losses})
    df.to_csv(fname)


if __name__ == "__main__":
    main()
