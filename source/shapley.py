"""
-------------------------------------------------------------------------------------------------------------

shapley.py, , v1.0 
by Oscar, February 2024

-------------------------------------------------------------------------------------------------------------

Federated Shapley implementation to work with Flower.

Implemented from the paper "A principled approach to data valuation for Federated Learning"
https://doi.org/10.48550/arXiv.2009.06192

Note:
The basic federated Shapley value is used for research purposes and accuracy of benchmarking. This 
requires a central test dataset at the aggregating server, this is not viable in many real cases of
federated learning. Implementation and testing of the following approaches should be considered in 
future work to enable contribution evaulation without requiring a central testset:
https://dl.acm.org/doi/10.14778/3587136.3587141 
https://proceedings.neurips.cc/paper_files/paper/2021/file/8682cc30db9c025ecd3fee433f8ab54c-Paper.pdf

Further:
Shapley calculations are computationally intensive and scales with O(T2^k) complexity for k clients
participating per round for T rounds. Approximations should be implemented for scaling but in this work,
a maximum of 5 clients participate in each round, which can be calculated in reasonable time.

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- Shapley - a class enabling full Shapley calculation in flower.

-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.shapley import Shapley
Instantiate with the central test set, the test function, set_parameters function and key config:
    >>> shap = Shapley(testloader, test, set_parameters, NUM_CLIENTS, Net().to(DEVICE))
Set parameters, w^t+1 and set the attributes corresponding to centralised evaluation results, this saves repeating 
the evaluation:
    >>> shap.aggregatedRoundParams = parameters
    >>> shap.f_o = accuracy
    >>> shap.centralLoss = loss
    >>> shap.round = server_round
Pass the client id's and corresponding parameters, w^t for the round in order to perform the Shapley calculation
and store the updated array of all results in the attribute resultsFedSV:
    >>> shap.fedSV(clients, parameters)
Interpret the contributions using:
    >>> contributions = shap.resultsFedSV

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
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#from torchvision.datasets import EMNIST
#from torch.utils.data import DataLoader, random_split
#import random
#from matplotlib import pyplot as plt
from math import comb
from itertools import combinations
#import flwr as fl
#from flwr.common import Metrics

class Shapley():
  """
  Implementation of the Federated Shapley value from:
  https://doi.org/10.48550/arXiv.2009.06192

  Attributes:
    dataset - the test set used to perform the evaluation
    test - the test function corresponding to the model/ dataset used.
    set_parameters - function required to set the nn params.
    num_clients - total number of clients in the federation
    resultsFedSV - holding all Shapley values in 2D array indexed by client id and round number.
    aggregatedRoundParams - the result of the aggregation, w^t+1
    centralLoss - as calculated in the centralised evaluation which is used given the central dataset
    round - the current round for indexing results
    f_o - the orchestrator fairness, stored for ease of use elsewhere.
    net - a nn.Module derived object defining an instance of the neural net/model.

  Methods:
    __utility_function - internal method to calculate the utility between a set of weights
    __power_set - internal method returning the improper power set from an input
    fedSV - the main entry point, handling full calculation of federated Shapley value

  """
  def __init__(self, test_set, test_function, set_parameters, number_clients, net):
    self.dataset = test_set
    self.test = test_function
    self.set_parameters = set_parameters
    self.num_clients = number_clients
    self.resultsFedSV = np.zeros(number_clients) # Holding the FedSV Shapley values indexed by client id
    self.aggregatedRoundParams = None # if required we can use: fl.common.ndarrays_to_parameters(get_parameters(Net()))
    self.centralLoss = 0 # Taken from the centralised evaluation as it is not necessary to compute twice
    self.round = 0 # Auto updated by the evaluation function for debugging
    self.f_o = 0
    self.net = net
    return

  def __utility_function(self, comparitive_weights):
    """
    Per round utility function.

    Input: array of model weights, if a nested array then average of the set's
           weights obtained.

    Output: the difference in the loss between the aggregated weights for the round
            and the average of the input set of weights.
    """
    # Perform an average on the comparitive weights
    set_size = len(comparitive_weights)
    if set_size == 1:
      alt_weights = comparitive_weights[0]
    else:
      alt_weights = np.mean([np.array(weights, dtype=object) for weights in comparitive_weights], axis=0)
    # Initialising a test net to compare to the centrally evaluated loss.
    self.set_parameters(self.net, list(alt_weights)) 
    loss, _, _ = self.test(self.net, self.dataset) # Testing on the central dataset
    return float(self.centralLoss - loss)

  def __power_set(self, input_set):
    """
    Returns the improper power set of the input set, the empty set is removed as
    it does not make sense to train a model on zero weights
    """
    input_list = list(input_set)
    power_set =  [list(c) for r in range(len(input_list) + 1) for c in combinations(input_list, r)]
    return power_set[1:]

  def fedSV(self, round_participants,
            participant_weights):
    """
    Calculating the original federated Shapley value.
    Method is called each round when the training has been completed and the
    server has aggregated parameters to form the new model.
    """
    print(f"Calculating the round {self.round} Shapley values")
    print(f"Round participants: {round_participants}")
    for i in range(self.num_clients):
      # if the client has not participated in training, skip, it is assigned
      # a Shapley value of zero for that round as per the paper:
      if i not in round_participants:
        continue
      else:
        # find the power set - all the subsets without client i:
        contributions = 0
        s_t = 0
        set_without_i = round_participants.copy()
        set_without_i.remove(i)
        power_set = self.__power_set(set_without_i)
        power_weights = [[participant_weights[j] for j in s] for s in power_set]
        for s in power_weights:
          without_i = self.__utility_function(s)
          t = s.copy()
          t.append(participant_weights[i])
          with_i = self.__utility_function(t)
          # We accumulate the contributions from each client for each round:
          contributions += (1 / comb(len(round_participants)-1,len(s))) * (with_i - without_i)
        s_t = contributions / len(round_participants)
        print(f"Client {i} has Shapley contribution {s_t}")
        self.resultsFedSV[i] = self.resultsFedSV[i] + s_t # updating the Shapley value
    return
