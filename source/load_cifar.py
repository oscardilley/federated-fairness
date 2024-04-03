"""
-------------------------------------------------------------------------------------------------------------

load_cifar.py, v1.0
by Oscar, March 2024

-------------------------------------------------------------------------------------------------------------

CIFAR 10 data loader and preprocessing using Flower Datasets.

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- apply_transforms - manipulates the data as required to be compatible with pytorch nn model.
- load_iid - iid loading of client trainloaders (90% split) and validation loaders (10%) as well as central testset.
- load_niid - same as above using Direchlet partioner, alpha = 0.5 to produce non-iid client datasets.


-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.load_cifar import load_niid, load_iid
Use either function to automatically generated loaders depending on number of clients and batch size:
    >>> trainloaders, valloaders, testloader, features = load_iid(NUM_CLIENTS, BATCH_SIZE)

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
#from collections import OrderedDict
#from typing import Dict, List, Optional, Tuple
#import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader#, random_split
# import random
# from matplotlib import pyplot as plt
# from math import comb
# from itertools import combinations
# import json
# from datetime import timedelta
# import time
# start = time.perf_counter()
# import flwr as fl
# from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_iid(num_clients, b_size):
    """ 
    Load iid split 

    Inputs:
        num_clients - the number of clients that require datasets
        b_size - the batch size used

    Outputs:
        trainloaders - a list of pytorch DataLoader for 90% train sets indexed by client.
        valloaders - a list of pytorch DataLoader for 10% test sets indexed by client.
        testloader - a single DataLoader for the centralised testset
        features - dataset information for displaying information.

    """
    # Download and transform CIFAR-10 (train and test)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients})
    # Loading the central testset:
    testset = fds.load_split("test") # central testset
    testloader = DataLoader(testset.with_transform(apply_transforms), batch_size=b_size)
    features = testset.features

    trainloaders = []
    valloaders = []
    # Looping through each client, splitting train into train and val and turning into Pytorch DataLoader
    for c in range(num_clients):
        partition = fds.load_partition(c)
        # Divide data on each node: 90% train, 10% validation
        partition_train_test = partition.train_test_split(test_size=0.1)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloaders.append(DataLoader(partition_train_test["train"], batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_train_test["test"], batch_size=b_size))
    return trainloaders, valloaders, testloader, features

def load_niid(num_clients, b_size):
    """ 
    Load niid split 
    
    Inputs:
        num_clients - the number of clients that require datasets
        b_size - the batch size used

    Outputs:
        trainloaders - a list of pytorch DataLoader for 90% train sets indexed by client.
        valloaders - a list of pytorch DataLoader for 10% test sets indexed by client.
        testloader - a single DataLoader for the centralised testset
        features - dataset information for displaying information.

    """
    # Statistical heterogeneity introduced using Dirichlet Partitioning:
    partitioner = DirichletPartitioner(num_partitions=num_clients, partition_by="label",
                                       alpha=0.5, min_partition_size=10,
                                       self_balancing=True)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    # Loading the central testset:
    testset = fds.load_split("test") # central testset
    testloader = DataLoader(testset.with_transform(apply_transforms), batch_size=b_size)
    features = testset.features

    trainloaders = []
    valloaders = []
    # Looping through each client, splitting train into train and val and turning into Pytorch DataLoader
    for c in range(num_clients):
        partition = fds.load_partition(c)
        # Divide data on each node: 90% train, 10% validation
        partition_train_test = partition.train_test_split(test_size=0.1)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloaders.append(DataLoader(partition_train_test["train"], batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_train_test["test"], batch_size=b_size))
    return trainloaders, valloaders, testloader, features