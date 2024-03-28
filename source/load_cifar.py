from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import random
from matplotlib import pyplot as plt
from math import comb
from itertools import combinations
import json
from datetime import timedelta
import time
start = time.perf_counter()
import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

def load_iid(num_clients, b_size):
    # Download and transform CIFAR-10 (train and test)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients})
    
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    testset = fds.load_split("test") # central testset
    testloader = DataLoader(testset.with_transform(apply_transforms), batch_size=b_size)
    features = testset.features

    trainloaders = []
    valloaders = []
    for c in range(num_clients):
        partition = fds.load_partition(c)
        # Divide data on each node: 90% train, 10% validation
        partition_train_test = partition.train_test_split(test_size=0.1)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloaders.append(DataLoader(partition_train_test["train"], batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_train_test["test"], batch_size=b_size))
    return trainloaders, valloaders, testloader, features

def load_niid(num_clients, b_size):
    # Download and transform CIFAR-10 (train and test)
    # Statistical heterogeneity introduced using Dirichlet Partitioning
    partitioner = DirichletPartitioner(num_partitions=num_clients, partition_by="label",
                                       alpha=0.5, min_partition_size=10,
                                       self_balancing=True)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    testset = fds.load_split("test") # central testset
    testloader = DataLoader(testset.with_transform(apply_transforms), batch_size=b_size)
    features = testset.features

    trainloaders = []
    valloaders = []
    for c in range(num_clients):
        partition = fds.load_partition(c)
        # Divide data on each node: 90% train, 10% validation
        partition_train_test = partition.train_test_split(test_size=0.1)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloaders.append(DataLoader(partition_train_test["train"], batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_train_test["test"], batch_size=b_size))
    return trainloaders, valloaders, testloader, features