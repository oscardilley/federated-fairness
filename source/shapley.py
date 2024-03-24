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

class Shapley():
  """
  Implementation of the different types of the Federated Shapley value.
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

  def __utility_function(self, comparitive_weights):#: List[fl.common.NDArrays]):
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
    self.set_parameters(self.net, list(alt_weights)) # issue with alt_weights averaging
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
    # NOTE - big issues around central dataset assumption - needs to be iid
    # and representative etc - all that I didn't want to have to do
    print(f"Calculating the round {self.round} Shapley values")
    print(f"Round participants: {round_participants}")
    for i in range(self.num_clients):
      # if the client has not participated in training, skip, it is assigned
      # a Shapley value of zero for that round:
      if i not in round_participants:
        continue
      else:
        # find the power set - all the subsets without client i
        contributions = 0
        s_t = 0
        set_without_i = round_participants.copy()
        set_without_i.remove(i)
        #print(f"We find the power set of {round_participants} without {i} which is {set_without_i}")
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
