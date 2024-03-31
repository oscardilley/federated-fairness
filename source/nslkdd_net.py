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
    # Inspired by https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/ 
    def __init__(self) -> None:
        super(Net, self).__init__() # Calls init method of Net superclass (nn.Module) enabling access to nn
        # Needs to start with input space as wide as preprocessed inputs, 123 wide including the class label
        self.layer1 = nn.Linear(122, 122, dtype=torch.float64)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(122, 122, dtype=torch.float64)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(122, 122, dtype=torch.float64)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(122, 1, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
        # Needs to end with 1 for binary classification


    def forward(self, x: torch.Tensor) -> torch.Tensor: # -> is an annotation for function output
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
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
        for data in trainloader:
            labels = (torch.Tensor([[x] for x in data["class"]]).float()).to(DEVICE)
            data.pop("class")
            inputs = torch.from_numpy(np.array([values.numpy() for key,values in data.items()], dtype=float)).to(DEVICE)
            inputs = inputs.mT # transpose required
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
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
        for data in testloader:
            labels = (torch.Tensor([[x] for x in data["class"]]).float()).to(DEVICE)
            data.pop("class")
            inputs = torch.from_numpy(np.array([values.numpy() for key,values in data.items()], dtype=float)).to(DEVICE)
            inputs = inputs.mT # transpose required
            outputs = net(inputs)
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

