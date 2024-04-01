from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
import pandas as pd
from math import inf

from datasets import Dataset

def load_iid(num_clients, b_size):
    # Download and transform CIFAR-10 (train and test)
    fds = FederatedDataset(dataset="Mireu-Lab/NSL-KDD", partitioners={"train": num_clients})
    fds_test = FederatedDataset(dataset="Mireu-Lab/NSL-KDD", partitioners={"test": 10})
    #testset = fds_test.load_partition(0) # central testset
    partition = fds.load_split("test") # using a reduced central testset
    testset = partition.train_test_split(test_size=0.01)["test"]
    features = testset.features
    # Grouping the categories by type of transform required:
    protocols = ['tcp','udp', 'icmp']
    services = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    categorical = {'protocol_type':protocols, 'service':services, 'flag':flags} # require label or one-hot encoding
    continuous =  ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    # set 'num_outbound_cmds': [0.0, 1.0] to avoid NaN
    cont_min_max = {'duration': [0.0, 57715.0], 'src_bytes': [0.0, 1379963888.0], 'dst_bytes': [0.0, 1309937401.0], 'land': [0.0, 1.0], 'wrong_fragment': [0.0, 3.0], 'urgent': [0.0, 3.0], 'hot': [0.0, 101.0], 'num_failed_logins': [0.0, 5.0], 'logged_in': [0.0, 1.0], 'num_compromised': [0.0, 7479.0], 'root_shell': [0.0, 1.0], 'su_attempted': [0.0, 2.0], 'num_root': [0.0, 7468.0], 'num_file_creations': [0.0, 100.0], 'num_shells': [0.0, 5.0], 'num_access_files': [0.0, 9.0], 'num_outbound_cmds': [0.0, 1.0], 'is_host_login': [0.0, 1.0], 'is_guest_login': [0.0, 1.0], 'count': [0.0, 511.0], 'srv_count': [0.0, 511.0], 'serror_rate': [0.0, 1.0], 'srv_serror_rate': [0.0, 1.0], 'rerror_rate': [0.0, 1.0], 'srv_rerror_rate': [0.0, 1.0], 'same_srv_rate': [0.0, 1.0], 'diff_srv_rate': [0.0, 1.0], 'srv_diff_host_rate': [0.0, 1.0], 'dst_host_count': [0.0, 255.0], 'dst_host_srv_count': [0.0, 255.0], 'dst_host_same_srv_rate': [0.0, 1.0], 'dst_host_diff_srv_rate': [0.0, 1.0], 'dst_host_same_src_port_rate': [0.0, 1.0], 'dst_host_srv_diff_host_rate': [0.0, 1.0], 'dst_host_serror_rate': [0.0, 1.0], 'dst_host_srv_serror_rate': [0.0, 1.0], 'dst_host_rerror_rate': [0.0, 1.0], 'dst_host_srv_rerror_rate': [0.0, 1.0]}
    label = 'class'
    classify = lambda x: 0 if x=='normal' else 1 # converted 'normal' to zero and 'abnormal' to 1

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # Runs when the data is accessed
        # https://towardsdatascience.com/deep-learning-using-pytorch-for-tabular-data-c68017d8b480
        # One hot encoding
        data = pd.DataFrame.from_dict(batch)
        data_onehot = pd.get_dummies(data, columns=categorical.keys(), prefix="", prefix_sep="")
        columns = data_onehot.columns.values.tolist()
        for labels in categorical.values(): # Ensuring all the columns are always there even if all zero
            for label in labels:
                if label not in columns:
                    data_onehot[label] = False
        # Normalisation of numerical columns:
        data_normalised = data_onehot.copy()
        for column in continuous:
            data_normalised[column] = ((data_normalised[column] - cont_min_max[column][0]) / (cont_min_max[column][1] - cont_min_max[column][0]))
        # Fix the label column to be binary 
        data_preprocessed = data_normalised.copy()
        data_preprocessed["class"] = [classify(x) for x in data_preprocessed["class"]]
        dataset = Dataset.from_pandas(data_preprocessed)
        return dataset

    # Preprocessing data for targets
    print("Testloading")
    testloader = DataLoader(apply_transforms(testset), batch_size=b_size)
    trainloaders = []
    valloaders = []
    print("Train/valloading")
    for c in range(num_clients):
        print(f"Preprocessing dataset for client {c}")
        partition = fds.load_partition(c)
        # Divide data on each node: 90% train, 10% validation
        partition_train_test = partition.train_test_split(test_size=0.01)
        partition_train = apply_transforms(partition_train_test["train"])
        partition_test = apply_transforms(partition_train_test["test"])
        trainloaders.append(DataLoader(partition_train, batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_test, batch_size=b_size))
        # if c < 1:
        # # Use to observe heterogeneity of sampling, will need to change after encoding
        #     data = trainloaders[c]
        #     temp = next(iter(data))["class"]
        #     normals = np.sum(np.array(temp) == 0)
        #     print(f" Normals = {normals}, Anomoly = {32 - normals}")
    return trainloaders, valloaders, testloader, features

def load_niid(num_clients, b_size):
    # Download and transform CIFAR-10 (train and test)
    # Statistical heterogeneity introduced using Dirichlet Partitioning
    partitioner = DirichletPartitioner(num_partitions=num_clients, partition_by="class",
                                       alpha=3, min_partition_size=10,
                                       self_balancing=True)
    fds = FederatedDataset(dataset="Mireu-Lab/NSL-KDD", partitioners={"train": partitioner})
    partition = fds.load_split("test") # using a reduced central testset
    testset = partition.train_test_split(test_size=0.05)["test"]
    features = testset.features
    # Grouping the categories by type of transform required:
    protocols = ['tcp','udp', 'icmp']
    services = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    categorical = {'protocol_type':protocols, 'service':services, 'flag':flags} # require label or one-hot encoding
    continuous = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    # set 'num_outbound_cmds': [0.0, 1.0] to avoid NaN
    cont_min_max = {'duration': [0.0, 57715.0], 'src_bytes': [0.0, 1379963888.0], 'dst_bytes': [0.0, 1309937401.0], 'land': [0.0, 1.0], 'wrong_fragment': [0.0, 3.0], 'urgent': [0.0, 3.0], 'hot': [0.0, 101.0], 'num_failed_logins': [0.0, 5.0], 'logged_in': [0.0, 1.0], 'num_compromised': [0.0, 7479.0], 'root_shell': [0.0, 1.0], 'su_attempted': [0.0, 2.0], 'num_root': [0.0, 7468.0], 'num_file_creations': [0.0, 100.0], 'num_shells': [0.0, 5.0], 'num_access_files': [0.0, 9.0], 'num_outbound_cmds': [0.0, 1.0], 'is_host_login': [0.0, 1.0], 'is_guest_login': [0.0, 1.0], 'count': [0.0, 511.0], 'srv_count': [0.0, 511.0], 'serror_rate': [0.0, 1.0], 'srv_serror_rate': [0.0, 1.0], 'rerror_rate': [0.0, 1.0], 'srv_rerror_rate': [0.0, 1.0], 'same_srv_rate': [0.0, 1.0], 'diff_srv_rate': [0.0, 1.0], 'srv_diff_host_rate': [0.0, 1.0], 'dst_host_count': [0.0, 255.0], 'dst_host_srv_count': [0.0, 255.0], 'dst_host_same_srv_rate': [0.0, 1.0], 'dst_host_diff_srv_rate': [0.0, 1.0], 'dst_host_same_src_port_rate': [0.0, 1.0], 'dst_host_srv_diff_host_rate': [0.0, 1.0], 'dst_host_serror_rate': [0.0, 1.0], 'dst_host_srv_serror_rate': [0.0, 1.0], 'dst_host_rerror_rate': [0.0, 1.0], 'dst_host_srv_rerror_rate': [0.0, 1.0]}
    label = 'class'
    classify = lambda x: 0 if x=='normal' else 1 # converted 'normal' to zero and 'abnormal' to 1

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # Runs when the data is accessed
        # https://towardsdatascience.com/deep-learning-using-pytorch-for-tabular-data-c68017d8b480
        # One hot encoding
        data = pd.DataFrame.from_dict(batch)
        data_onehot = pd.get_dummies(data, columns=categorical.keys(), prefix="", prefix_sep="")
        columns = data_onehot.columns.values.tolist()
        for labels in categorical.values(): # Ensuring all the columns are always there even if all zero
            for label in labels:
                if label not in columns:
                    data_onehot[label] = False
        # Normalisation of numerical columns:
        data_normalised = data_onehot.copy()
        for column in continuous:
            data_normalised[column] = ((data_normalised[column] - cont_min_max[column][0]) / (cont_min_max[column][1] - cont_min_max[column][0]))
        # Fix the label column to be binary 
        data_preprocessed = data_normalised.copy()
        data_preprocessed["class"] = [classify(x) for x in data_preprocessed["class"]]
        dataset = Dataset.from_pandas(data_preprocessed)
        return dataset

    # Preprocessing data for targets
    print("Testloading")
    testloader = DataLoader(apply_transforms(testset), batch_size=b_size)
    trainloaders = []
    valloaders = []
    print("Train/valloading")
    for c in range(num_clients):
        print(f"Preprocessing dataset for client {c}")
        partition = fds.load_partition(c)
        # Divide data on each node: 90% train, 10% validation
        partition_train_test = partition.train_test_split(test_size=0.1)
        partition_train = apply_transforms(partition_train_test["train"])
        partition_test = apply_transforms(partition_train_test["test"])
        trainloaders.append(DataLoader(partition_train, batch_size=b_size, shuffle=True))
        valloaders.append(DataLoader(partition_test, batch_size=b_size))
        # if c < 1:
        # # Use to observe heterogeneity of sampling, will need to change after encoding
        #     data = trainloaders[c]
        #     temp = next(iter(data))["class"]
        #     normals = np.sum(np.array(temp) == 0)
        #     print(f" Normals = {normals}, Anomoly = {32 - normals}")
    return trainloaders, valloaders, testloader, features

def min_max(num_clients):
    # Parsing the dataset to establish the minimum and maximum values in each numerical field
    partitioner = DirichletPartitioner(num_partitions=num_clients, partition_by="class",
                                       alpha=3, min_partition_size=10,
                                       self_balancing=True)
    fds = FederatedDataset(dataset="Mireu-Lab/NSL-KDD", partitioners={"train": partitioner})
    # Loading the test and train
    trainset = fds.load_split("train")
    testset = fds.load_split("test")
    # Extracting labels
    features = testset.features
    # Grouping the categories by type of transform required:
    protocols = ['tcp','udp', 'icmp']
    services = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    categorical = {'protocol_type':protocols, 'service':services, 'flag':flags} # require label or one-hot encoding
    continuous = list(features.keys()) # require normalisation
    continuous.remove('class')
    for i in categorical:
        continuous.remove(i)
    cont_min_max = {}
    for c in continuous:
        cont_min_max[c] = [inf,0]
    # Parsing trainset:
    counter = 0
    train = iter(trainset)
    while True:
        try:
            row = next(train)
            for c in continuous:
                val = float(row[c])
                # issue on first iteration 
                if (val < cont_min_max[c][0]):
                    cont_min_max[c][0] = val # reassigning min
                if (val > cont_min_max[c][1]):
                    cont_min_max[c][1] = val # reassigning min
            counter += 1
            if counter % 10000 == 0:
                print(f"Processed {counter} rows")
        except StopIteration:
            print("Reached end of trainset")
            break
    print(f"Processed {counter} rows")
    # Parsing testset
    counter = 0
    test = iter(testset)
    while True:
        try:
            row = next(test)
            for c in continuous:
                val = float(row[c])
                if (val < cont_min_max[c][0]):
                    cont_min_max[c][0] = val # reassigning min
                if (val > cont_min_max[c][1]):
                    cont_min_max[c][1] = val # reassigning min
            counter += 1
            if counter % 10000 == 0:
                print(f"Processed {counter} rows")
        except StopIteration:
            print("Reached end of testset")
            break
    print(f"Processed {counter} rows")
    return cont_min_max
    

# minmax = min_max(100) # generate the min max value dictionary
#trainloaders, valloaders, testloader, _ = load_niid(200, 32)
