"""
-------------------------------------------------------------------------------------------------------------

client.py, , v1.0
by Oscar, March 2024

-------------------------------------------------------------------------------------------------------------

Custom Flower client instance to handle requirements of different frameworks for fairness testing.
See below for the abstract base class for the Flower client.
https://github.com/adap/flower/blob/94204242a737368926e0e48342ffe025dc7b3409/src/py/flwr/client/client.py

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- DEVICE - automatically detects whether training on cuda or cpu.
- FlowerClient - implements the client's model fitting behaviour when called by strategy.
- get_parameters - Takes state dict values to numpy to be processed by Flower.
- set_parameters - converts numpy parameters processed by flower back to state dict values.

-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.client import FlowerClient, DEVICE, get_parameters, set_parameters
Instantiate client by passing client id (cid), model (net), trainloader, valloader, test and train functions: 
    >>> FlowerClient(cid, net, trainloader, valloader, test, train)
Use get and set parameter functions for passing parameters to and from the Flower strategy.

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
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets import EMNIST
# from torch.utils.data import DataLoader, random_split
#import random
#from matplotlib import pyplot as plt
#from math import comb
#from itertools import combinations
import flwr as fl
from flwr.common import Metrics

# Simulation configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu") # use for debugging when RuntimeError: CUDA error: device-side assert triggered

class FlowerClient(fl.client.NumPyClient):
    """
    Defines the client behaviour for FedAvg, q-FedAvg, Ditto and FedMinMax strategies.
    Implements fairness measurement and federated evaluation for each round a client is selected.

    Attributes:
        cid - client id for dynamic client loading without occupying excessive memory.
        net - a nn.Module derived object defining an instance of the neural net/model.
        temp - a copy of net used for data formatting in ditto implementation.
        trainloader - pytorch DataLoader object containing the local train dataset.
        valloader - pytorch DataLoader object containing the local test dataset.
        test - bespoke test function for the dataset used, defined in *_net.py files.
        train - bespoke train function for the dataset used, defined in *_net.py files.
        per_params - parameter attribute for storing personalised parameters for Ditto.
        risks - the risks per protected group for the FedMinMax implementation.

    Methods:
        get_parameters - returns the current parameters of self.net
        fit - detects the strategy used, obtains strategy parameters, trains the model and 
            gathers metrics for fairness analytics.
    """
    def __init__(self, cid, net, trainloader, valloader, test_function, train_function, per_params=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.test = test_function
        self.train = train_function
        self.per_params = per_params
        self.risks = None
        # Checking client dataset length without assuming batch size
        dataset_length = 0
        for data in trainloader:
            if "label" in data:
                labels = data["label"]
            else:
                labels = data["class"] # accounts for NSL-KDD
            dataset_length += len(labels)
        for data in valloader:
            if "label" in data:
                labels = data["label"]
            else:
                labels = data["class"] # accounts for NSL-KDD
            dataset_length += len(labels)
        self.dataset_len = dataset_length

    def get_parameters(self, config):
        """ Return the parameters of self.net """
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """
        Obtain strategy information, orchestrates training and collects data.

        Inputs:
        parameters - the new set of parameters from the aggregating server.
        config - a dictionary passed from the strategy indicating the strategy's characteristics.

        Outputs: 
            params - updated parameters after E local epochs.
            len(self.trainloader)
            {...} a dict containing key measurements required for fairness calculation and strategy behaviour.
        """
        # Unpacking config parameters:
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        sensitive_attributes = config["sensitive_attributes"]
        # Detecting the strategy on the first round and setting a flag:
        if self.per_params == None:
            if "ditto" in config: # Initialising personalised params to be global
                print(f"Client {self.cid} pers param initialisation")
                self.per_params = parameters
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        # Training and evaluating
        set_parameters(self.net, parameters)
        # Determining the reward the client has receieved from the server for incentive fairness in the case it is the model itself:
        _, reward, _ = self.test(self.net, self.valloader, []) 
        # Default is FedAvg:
        if "ditto" in config:
            # Training and storing the parameters at the end of training.
            self.train(self.net, self.trainloader, epochs=local_epochs)
            params = get_parameters(self.net)
            ditto_parameters = config["ditto"]
            set_parameters(self.net, self.per_params)
            opts = {"opt": "ditto", "lambda": ditto_parameters["lambda"], "eta": ditto_parameters["eta"], "global_params": params}
            self.train(self.net, self.trainloader, epochs=int(ditto_parameters["s"]), option=opts)                                                         
            # Updating personalised params stored:
            self.per_params = get_parameters(self.net)
        elif "fedminmax" in config:
            opts = config["fedminmax"]
            returns = self.train(self.net, self.trainloader, epochs=local_epochs, option = opts)
            params = get_parameters(self.net)
            self.risks = returns["fedminmax_risks"] # is return to the server to update weighing coefficients
        else: # Default case
            # Training and storing the parameters at the end of training.
            self.train(self.net, self.trainloader, epochs=local_epochs)
            params = get_parameters(self.net)
        # Performing federated evaluation on the clients that are sampled for training:
        print(f"[Client {self.cid}] evaluate, config: {config}")
        loss, accuracy, group_eod = self.test(self.net, self.valloader, sensitive_attributes)
        group_fairness = dict(zip(sensitive_attributes, group_eod))

        return params, len(self.trainloader), {"cid":int(self.cid), 
                                               "personal_parameters": self.per_params,
                                               "parameters": params, 
                                               "accuracy": float(accuracy), 
                                               "loss": float(loss), 
                                               "group_fairness": group_fairness, 
                                               "reward": float(reward), 
                                               "risks": self.risks}

def get_parameters(net) -> List[np.ndarray]:
    """taking state_dict values to numpy (state_dict holds learnable parameters) """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """ Setting the new parameters in the state_dict from numpy that flower operated on """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
