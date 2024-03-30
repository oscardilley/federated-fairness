from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, random_split
import random
from matplotlib import pyplot as plt
from math import comb
from itertools import combinations
import flwr as fl
from flwr.common import Metrics
# Local import
from .client import DEVICE

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__() # Calls init method of Net superclass (nn.Module) enabling access to nn
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # -> is an annotation for function output
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    # taking state_dict values to numpy (state_dict holds learnable parameters)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    # Setting the new parameters in the state_dict from numpy that flower operated on
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for i, data in enumerate(trainloader,0):
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, sensitive_labels=[]):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    group_performance = [[0,0] for label in range(len(sensitive_labels))] # preset for EOP calc, will store the performance
    # init array for storing EOP information
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Cycles through in batches, set in data loader as 32
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            # Comparing the predicted to the inputs in order to determine EOP
            matched = (predicted == labels)
            for label in range(len(sensitive_labels)):
              labelled = (labels == label)
              not_labelled = (labels != label)
              group_performance[label][0] += (matched == labelled).sum()
              group_performance[label][1] += (matched == not_labelled).sum()
            total += labels.size(0)
            correct += matched.sum().item()
    for index in range(len(group_performance)):
      # Calculating P(Y.=1|A=1,Y=1) - P(Y.=1|A=0,Y=1) for each:
      # NB: could expand EOP to EOD by accounting for all results not just the correct results, seeing if predictions match
      group_performance[index] = float((group_performance[index][0] - group_performance[index][1]) / total)
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy, group_performance

