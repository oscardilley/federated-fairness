"""
-------------------------------------------------------------------------------------------------------------

nslkdd_net.py, , v1.0
by Oscar, April 2024

-------------------------------------------------------------------------------------------------------------

A binary classification model for the NSL-KDD network intrusion detection dataset.

Note: the aim of this project is not model optimisation, well respected baseline models have been selected
such that the development time can be spend on fairness analytics. 

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- Net - a nn.Module derived class defining the architecture of the neural net/model.
- train - a training function using BinaryCrossEntropyLoss and the Adam optimiser.
- test - testing and evaluating on a separate testset and gathering collecting data on the protected group
    performance.

-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.nslkdd_net import Net, train, test
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
import numpy as np
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
    A linear model consisting of four fully connected layers:
    Design influenced by: https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/ 
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        # Needs to start with input space as wide as preprocessed inputs, 123 wide including the class label
        self.layer1 = nn.Linear(122, 122, dtype=torch.float64)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(122, 122, dtype=torch.float64)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(122, 122, dtype=torch.float64)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(122, 1, dtype=torch.float64) # ends with single output binary classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ A forward pass through the network. """
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
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
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for data in trainloader:
            labels = (torch.Tensor([[x] for x in data["class"]]).double()).to(DEVICE)
            data.pop("class")
            inputs = torch.from_numpy(np.array([values.numpy() for key,values in data.items()], dtype=float)).to(DEVICE)
            inputs = inputs.mT # transpose required
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(1), labels.squeeze(1))
            loss.backward()
            if option is not None:
                if option["opt"] == "ditto":
                    ditto_manual_update(option["eta"], option["lambda"], option["global_params"])
            else:
                optimizer.step()
            # Train metrics:
            epoch_loss += loss
            total += labels.size(0)
            correct += (outputs.round() == labels).sum().item()
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
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    group_performance = [[0,0] for label in range(len(sensitive_labels))] # preset for EOP calc, will store the performance
    net.eval()
    with torch.no_grad():
        for data in testloader:
            labels = (torch.Tensor([[x] for x in data["class"]]).double()).to(DEVICE)
            data.pop("class")
            inputs = torch.from_numpy(np.array([values.numpy() for key,values in data.items()], dtype=float)).to(DEVICE)
            inputs = inputs.mT # transpose required
            outputs = net(inputs)
            loss += criterion(outputs.squeeze(1), labels.squeeze(1)).item()
            predicted = outputs.round()
            # Comparing the predicted to the inputs in order to determine EOD
            matched = (predicted == labels)
            for label in range(len(sensitive_labels)):
              labelled = (labels == label)
              not_labelled = (labels != label)
              group_performance[label][0] += (matched == labelled).sum()
              group_performance[label][1] += (matched == not_labelled).sum()
            total += labels.size(0)
            correct += matched.sum().item()
    for index in range(len(group_performance)):
        # Calculating EOD: P(Y.=1|A=1,Y=y) - P(Y.=1|A=0,Y=y) for each:
        group_performance[index] = float((group_performance[index][0] - group_performance[index][1]) / total)
    loss /= len(testloader.dataset)
    accuracy = correct / total

    return loss, accuracy, group_performance

