"""
-------------------------------------------------------------------------------------------------------------

femnist_net.py, , v1.0
by Oscar, March 2024

-------------------------------------------------------------------------------------------------------------

A FEMNIST classification CNN Pytorch model.

Note: the aim of this project is not model optimisation, well respected baseline models have been selected
such that the development time can be spend on fairness analytics. 

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- Net - a nn.Module derived class defining the architecture of the neural net/model.
- train - a training function using CrossEntropyLoss and the Adam optimiser.
- test - testing and evaluating on a separate testset and gathering collecting data on the protected group
    performance.

-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.femnist_net import Net, train, test
Instantiate:
    >>> net = Net().to(DEVICE)
Gather initial parameters if required:
    >>> get_parameters(Net())

-------------------------------------------------------------------------------------------------------------

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-------------------------------------------------------------------------------------------------------------
"""
# from collections import OrderedDict
# from typing import Dict, List, Optional, Tuple
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets import EMNIST
# from torch.utils.data import DataLoader, random_split
# import random
# from matplotlib import pyplot as plt
# from math import comb
# from itertools import combinations
# import flwr as fl
# from flwr.common import Metrics
# Local import
from .client import DEVICE

class Net(nn.Module):
    """
    A CNN consisting of (in order):
        Two 2D convolutional layers, conv1 and conv2 each using leaky relu and followed by maxpooling layer.
        A linear layer with dropout, fcon1
        A fully connected linear layer to output into 62 bins corresponding to the 62 FEMNIST classes.
    """
    def __init__(self) -> None:
        super(Net, self).__init__() 
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ A forward pass through the network. """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x

def train(net, trainloader, epochs: int, option = None):
    """
    Train the network on the training set.
    
    Inputs:
        net - the instance of the model
        trainloader - a pytorch DataLoader object.
        epochs - the number of local epochs to train over
        option - a flag to enable alternative training regimes such as ditto
    """
    def ditto_manual_update(lr, lam, glob):
        """ Manual parameter updates for ditto """
        with torch.no_grad():
            counter = 0
            q = [torch.from_numpy(g).to(DEVICE) for g in glob]
            for p in net.parameters():
                new_p = p - lr*(p.grad + (lam * (p - q[counter])))
                p.copy_(new_p)
                counter += 1
            return
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
                    ditto_manual_update(option["eta"], option["lambda"], option["global_params"])
            else:
                optimizer.step()
            # Train metrics:
            epoch_loss += loss
            total += length
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, sensitive_labels=[]):
    """
    Evaluate the network on the inputted test set and determine the equalised odds for each protected group.
    
    Inputs:
        net - the instance of the model
        testloader - a pytorch DataLoader object.
        sensitive_labels - a list of the class indexes associated with the protected groups in question.

    Outputs:
        loss - average loss 
        accuracy - accuracy calculated as the number of correct classificatins out of the total
        group_performance - a list of equalised odds measurers for each protected group given.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    group_performance = [[0,0] for label in range(len(sensitive_labels))] # preset for EOP calc, will store the performance
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
            # Comparing the predicted to the inputs in order to determine EOD
            matched = (predicted == labels)
            for label in range(len(sensitive_labels)):
                labelled = (labels == label)
                not_labelled = (labels != label)
                group_performance[label][0] += (matched == labelled).sum()
                group_performance[label][1] += (matched == not_labelled).sum()
            total += length
            correct += matched.sum().item()
    for index in range(len(group_performance)):
        # Calculating EOD: P(Y.=1|A=1,Y=y) - P(Y.=1|A=0,Y=y) for each:
        group_performance[index] = float((group_performance[index][0] - group_performance[index][1]) / total)
    loss /= len(testloader.dataset)
    accuracy = correct / total

    return loss, accuracy, group_performance

