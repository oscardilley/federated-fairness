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
        self.fmaps1 = 40
        self.fmaps2 = 160
        self.dense = 200
        self.dropout = 0.4
        self.batch_size = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.fmaps1, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.fmaps1, out_channels=self.fmaps2, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(49*self.fmaps2, self.dense), nn.LeakyReLU())
        self.fcon2 = nn.Linear(self.dense, 62)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # -> is an annotation for function output
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    # taking state_dict values to numpy (state_dict holds learnable parameters)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    # Setting the new parameters in the state_dict from numpy that flower operated on
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int, option = None):
    """Train the network on the training set."""
    ditto_update = lambda p, lr, grad, lam, pers, glob: p - lr*(grad + (lam *(pers - glob)))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images = batch['pixels']
            labels = batch['label']
            length = len(images)
            images = torch.reshape(images, (length, 1, 28, 28)) # required to meet NN input shape
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels.long())
            loss.backward()
            if option is not None:
                if option["opt"] == "ditto":
                    # Ditto regularisation
                    #print(f"Per Params are of type {type(option["per_params"])} and have size {option["per_params"].size()}")
                    #print(f"Params are of type {type(option["params"])} and have size {option["params"].size()}")
                    # might need to flatten, and reshape
                    # might need to form state dict like in set_parameters
                    # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/15
                    
                    reg_term = option["lambda"]*(np.array(option["per_params"]) - np.array(option["params"]))

                    # Learning rate should be accounted for elsewhere
                    with torch.no_grad():
                        for p in model.parameters():
                            # how do we get the correct params etc 
                            # is the personal the same as in the model?
                            new_p = ditto_update(p, , p.grad, , ,)
                            p.copy_(new_p)
                else:
                    optimizer.step()

            
        
            # Metrics
            epoch_loss += loss
            total += length
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
        for batch in testloader:
            images = batch['pixels']
            labels = batch['label']
            length = len(images)
            images = torch.reshape(images, (length, 1, 28, 28)) # required to meet NN input shape
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels.long()).item()
            _, predicted = torch.max(outputs.data, 1)
            # Comparing the predicted to the inputs in order to determine EOP
            matched = (predicted == labels)
            for label in range(len(sensitive_labels)):
              labelled = (labels == label)
              not_labelled = (labels != label)
              group_performance[label][0] += (matched == labelled).sum()
              group_performance[label][1] += (matched == not_labelled).sum()
            total += length
            correct += matched.sum().item()
    for index in range(len(group_performance)):
      # Calculating P(Y.=1|A=1,Y=1) - P(Y.=1|A=0,Y=1) for each:
      # NB: could expand EOP to EOD by accounting for all results not just the correct results, seeing if predictions match
        group_performance[index] = float((group_performance[index][0] - group_performance[index][1]) / total)
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy, group_performance

