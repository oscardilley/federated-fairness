"""
-------------------------------------------------------------------------------------------------------------

cifar_net.py, , v1.0
by Oscar, March 2024

-------------------------------------------------------------------------------------------------------------

A CIFAR-10 classification CNN Pytorch model. Structure following flower tutorial:
https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html 

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
    >>> from source.cifar_net import Net, train, test
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
from torch.utils.data import DataLoader
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
        A 2D convolutional layer conv1 
        A 2D maxpooling layer pool
        A further 2D convolutional layer conv2
        3 fully connected linear layers, fc1, fc2, fc3 outputing a classification into one of ten bins.
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ A forward pass through the network. """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    sum_risks = None # Placeholder for fedminmax return
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

    def fedminmax_manual_update(lr, risk):
        """ Manual parameter updates for FedMinMax strategy"""
        with torch.no_grad():
            for p in net.parameters():
                new_p = p - (lr*(p.grad)*risk)
                p.copy_(new_p)
            return

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for i, data in enumerate(trainloader,0):
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            batch_size = len(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward() # calculated the gradients
            if option is not None:
                if option["opt"] == "ditto":
                    # Ditto personalised updates, used https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/15
                    ditto_manual_update(option["eta"], option["lambda"], option["global_params"])
                if option["opt"] == "fedminmax":
                    sensitive_attributes = option["attributes"]
                    if sum_risks is None:
                        sum_risks = np.array([0.0 for a in sensitive_attributes])
                    subset_losses = [0 for a in sensitive_attributes]
                    subsets = [[] for a in sensitive_attributes]
                    number_sensitive = [0 for group in sensitive_attributes]
                    # Building mini datasets for each sensitive attribute:
                    for l in range(batch_size):
                        try:
                            i = sensitive_attributes.index(int(labels[l]))
                        except:
                            continue
                        subsets[i].append(images[l])
                        number_sensitive[i] += 1
                    # Testing the mini datasets to determine the risks:
                    for s in range(len(sensitive_attributes)):
                        if subsets[s] == []:
                            continue
                        lbls = torch.Tensor([sensitive_attributes[s] for img in range(len(subsets[s]))]).long().to(DEVICE)
                        imgs = (torch.stack(subsets[s])).to(DEVICE)
                        subset_loss = criterion(net(imgs), lbls)
                        subset_losses[s] += float(subset_loss)
                    # Calculating the key risk parameters
                    risks = np.nan_to_num(np.array(subset_losses))
                    sum_risks += risks
                    risk = np.sum((np.array(option["w"]) * risks) / batch_size)
                    fedminmax_manual_update(option["lr"], risk)
            else:
                optimizer.step()
            # Train metrics:
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return {"fedminmax_risks": sum_risks}



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
    group_performance = [[0,0] for label in range(len(sensitive_labels))] # preset for EOD calc, will store the performance
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Cycles through in batches
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
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

