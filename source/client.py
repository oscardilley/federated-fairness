"""

"""
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

# Simulation configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, test_function, train_function):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.test = test_function
        self.train = train_function

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # The training method for the client
        # Need to use the config dictionary in order to
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        sensitive_attributes = config["sensitive_attributes"]
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        _, reward, _ = self.test(self.net, self.valloader, []) # The reward is defined as the accuracy of the model on the central data
        self.train(self.net, self.trainloader, epochs=local_epochs)
        # Performing federated evaluation on the clients that are sampled for training:
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, get_parameters(self.net)) # getting end of training round accuracy
        loss, accuracy, group_eod = self.test(self.net, self.valloader, sensitive_attributes)
        # Need to process the EOD data here to determine group fairness:
        group_fairness = dict(zip(sensitive_attributes, group_eod))
        # Can include enhanced processing to show which groups are not performing well (eop's further from zero)
        return get_parameters(self.net), len(self.trainloader), {"cid":int(self.cid), "parameters": get_parameters(self.net), "accuracy": float(accuracy), "loss": float(loss), "group_fairness": group_fairness, "reward": float(reward)}

def get_parameters(net) -> List[np.ndarray]:
    # taking state_dict values to numpy (state_dict holds learnable parameters)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    # Setting the new parameters in the state_dict from numpy that flower operated on
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
