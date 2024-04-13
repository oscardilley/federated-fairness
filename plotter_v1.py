"""
-------------------------------------------------------------------------------------------------------------

ploter_v1.py, , v1.0
by Oscar, April 2024

-------------------------------------------------------------------------------------------------------------

MatPlotLib Plotting Script

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- 

-------------------------------------------------------------------------------------------------------------

Usage:


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

import matplotlib.pyplot as plt
import numpy as np
import json
import sys


def main(strategy, dataset, setting, clients, epochs):
  """ Main entry point"""
  # Config information - ensures plot annotations are correct:
  STRATEGY = strategy # "Ditto"/"FedAvg"/"q_FedAvg"/"FedMinMax"
  DATASET = dataset # "FEMNIST"/ "CIFAR"/ "NSLKDD"
  SETTING = setting # "niid"/"iid"
  CLIENTS = clients # 100/10/205
  PROPORTION = int((5 / CLIENTS) * 100)
  EPOCHS = epochs # 5 for FEMNIST and NSLKDD, 10 for CIFAR
  ROUNDS = 30
  NUMBER_REPEATS = 3
  r=15 - 1 # defines round range to display on bar chart
  # initialising data structures
  data_structure = {
    "rounds": [],
    "general_fairness": {
          "f_j": [],
          "f_g": [],
          "f_r": [],
          "f_o": []},
      "config": {
          "num_clients": 0,
          "local_epochs": 0,
          "num_rounds": 0,
          "batch_size": 0,
          "selection_rate": 0,
          "sensitive_attributes": []},
      "per_client_data": {
            "shap": [],
            "accuracy": [],
            "avg_eop": [],
            "gains": []}}
  path_extensions = [f'{STRATEGY}_{DATASET}_{SETTING}_{CLIENTS}C_{PROPORTION}PC_{EPOCHS}E_{ROUNDS}R_v{i+1}.json' for i in range(NUMBER_REPEATS)]
  temp = [None for i in range(NUMBER_REPEATS)]

  for repeat in range(NUMBER_REPEATS):
      with open("./Results/" + path_extensions[repeat], "r") as openfile:
          temp[repeat] = json.load(openfile) # Imported the JSON as a dictionary
      # Average the results and store in data
  data = averager(temp, data_structure)
  figure_path = f'./Results/Plots/{STRATEGY}_{DATASET}_{SETTING}_{CLIENTS}C'
  # Initialising plotter:
  plotter = FairnessEval(rounds = data["rounds"],
                        f_j = data["general_fairness"]["f_j"],
                        f_g = data["general_fairness"]["f_g"],
                        f_r = data["general_fairness"]["f_r"],
                        f_o = data["general_fairness"]["f_o"],
                        plot_string = f"{STRATEGY}, {DATASET}, {SETTING}, $\mu_s={round((5 / CLIENTS) * 100 , 1)}$%, $C={CLIENTS}$"
                        )
  # Producing Plots
  plotter.bar(figure_path, r)
  plotter.timeSeries(figure_path)
  return 

def averager(data, skeleton):
    """Cleaning and computing the average of a number of repeats of identically formatted data"""
    l = len(data)
    def average_dicts(d, s):
        """ """
        output = s # initalise the output empty
        for key, val in s.items():
            if type(val) is dict:
                if key == "config": # don't average config dict, set as any of the configs
                    output[key] = d[0][key]
                elif key == "per_client_data":
                    continue # skipping processing this as not necessary currently
                else:
                    # going down to the next layer
                    output[key] = average_dicts([b[key] for b in d], val)
            else: # assumed list
                output[key] = [np.mean(np.array([a[key][i] for a in d])) for i in range(len(d[0][key]))]
        return output



      # need to keep config unchanged
    
    return average_dicts(data, skeleton)




class FairnessEval():
  """
  A class designed to extract and display the fairness data analysed for the FL system.





  """
  def __init__(self,*, rounds=None, f_j=None, f_g=None, f_r=None, f_o=None, plot_string = "<enter here>"):
      self.f_j = f_j
      self.f_g = f_g
      self.f_r = f_r
      self.f_o = f_o
      self.rounds = rounds
      self.colour = ["#ef476f", "#073b4c", "#06d6a0", "#118ab2", "#ffd166"]
      self.labels = ["Individual", "Protected Group", "Incentive", "Orchestrator"]
      self.weights = np.array([0.2, 0.2, 0.2, 0.2])
      self.plot_string = plot_string
      return

  def bar(self, path, r):
      # Plotting a bar chart of the average with error bars
      fig, ax = plt.subplots(1)
      fig.suptitle(f"Average Fairness From Round {r+1} to {int(max(self.rounds)+1)}", fontsize = 15)
      ax.grid(axis = "y", linewidth = 0.5, linestyle = "--")
      ax.set_axisbelow(True)
    #   metrics = [np.mean(np.array(self.f_j)),np.mean(np.array(self.f_g)),np.mean(np.array(self.f_r)),max(self.f_o)]
    #   f_max = lambda f: max(f) - np.mean(np.array(f))
    #   f_min = lambda f: np.mean(np.array(f)) - min(f)
    #   iqr_low = lambda f: np.mean(np.array(f)) - np.percentile(f, 25)
    #   iqr_high = lambda f: np.percentile(f, 75) - np.mean(np.array(f))
      # ax.bar(self.labels, metrics, color=[self.colour[0], self.colour[2], self.colour[3], self.colour[4]], linewidth = 0, alpha = 0.7)
      # ax.bar(self.labels, metrics, edgecolor = [self.colour[0], self.colour[2], self.colour[3], self.colour[4]], linewidth = 2, fill = False)
      # ax.errorbar(self.labels, metrics, yerr=[[f_min(self.f_j),f_min(self.f_g),f_min(self.f_r),f_min(self.f_o)],
      #                                    [f_max(self.f_j),f_max(self.f_g),f_max(self.f_r),f_max(self.f_o)]],
      #                                    fmt = ".r", capsize = 3)
      # ax.errorbar(self.labels, metrics, yerr=[[iqr_low(self.f_j),iqr_low(self.f_g),iqr_low(self.f_r),iqr_low(self.f_o)],
      #                                    [iqr_high(self.f_j),iqr_high(self.f_g),iqr_high(self.f_r),iqr_high(self.f_o)]],
      #                                    fmt = "ok", capsize = 12)
      # ax.errorbar(self.labels, metrics, yerr=[[iqr_low(self.f_j),iqr_low(self.f_g),iqr_low(self.f_r),iqr_low(self.f_o)],
      #                                    [iqr_high(self.f_j),iqr_high(self.f_g),iqr_high(self.f_r),iqr_high(self.f_o)]],
      #                                    fmt = "ok", capsize = 12)
      general_fairness = np.mean(np.array([self.f_j[r:], self.f_g[r:], self.f_r[r:], self.f_o[r:]]))
      ax.axhline(general_fairness, linewidth = 1.5, color='k', linestyle='--')
      ax.annotate(f"$F_T$={round(general_fairness,2)}", [3.29, general_fairness + 0.015])
      box = ax.boxplot([self.f_j[r:], self.f_g[r:], self.f_r[r:], self.f_o[r:]], notch = False, patch_artist=True, labels=self.labels, sym="+")
      for median, whisker, cap in zip(box['medians'], box['whiskers'], box['caps']):
          median.set_color("k")
          whisker.set_color("k")
          cap.set_color("k")
          median.set_linewidth(1.3)
          whisker.set_linewidth(1.3)
          cap.set_linewidth(1.3)
      for patch, colour in zip(box['boxes'], [self.colour[0], self.colour[2], self.colour[3], self.colour[4]]):
          patch.set_facecolor(colour)
          patch.set_linewidth(1.3)
      ax.set_xlabel("Fairness Notions", fontsize = 12)
      ax.set_ylabel("Normalised Fairness", fontsize = 12)
      ax.set_title(self.plot_string, fontsize = 12)
      ax.set_ylim([0,1.1])
      plt.gcf().savefig(path + '_bar_v2.png', dpi = 400)
      return

  def timeSeries(self, path):
      # Plotting the values of each metric against the round
      fig,ax = plt.subplots(1)
      fig.suptitle("Timeseries Fairness Evaluation", fontsize = 15)
      ax.grid(linewidth = 0.5, linestyle = "--")
      ax.set_axisbelow(True)
      general_fairness = [np.mean(np.array([self.f_j[i], self.f_g[i], self.f_r[i], self.f_o[i]])) for i in range(len(self.rounds))]
      ax.scatter(self.rounds, self.f_j, color = self.colour[0], linewidth = 0.8)
      ax.scatter(self.rounds, self.f_g, color = self.colour[2], linewidth = 0.8)
      ax.scatter(self.rounds, self.f_r, color = self.colour[3], linewidth = 0.8)
      ax.scatter(self.rounds, self.f_o, color = self.colour[4], linewidth = 0.8)
      ax.plot(self.rounds, general_fairness, color = self.colour[1], linewidth = 2.2)
      ax.plot(self.rounds, self.f_j, color = self.colour[0], linewidth = 0.8)
      ax.plot(self.rounds, self.f_g, color = self.colour[2], linewidth = 0.8)
      ax.plot(self.rounds, self.f_r, color = self.colour[3], linewidth = 0.8)
      ax.plot(self.rounds, self.f_o, color = self.colour[4], linewidth = 0.8)
      values = {"0": np.mean(np.array(self.f_j)), "1": np.mean(np.array(self.f_g)), "3": np.mean(np.array(self.f_r)), "4": np.mean(np.array(self.f_o))}
      layers = sorted(values.items(), key=lambda item:item[1]) # to get the background layering correct
      data = [self.f_j, self.f_g, self.f_r, self.f_o]
      # ax.fill_between(self.rounds, data[int(layers[4][0])], color = self.colour[int(layers[4][0])], alpha = 0.3)
      # ax.fill_between(self.rounds, data[int(layers[3][0])], color = self.colour[int(layers[3][0])], alpha = 0.3)
      # ax.fill_between(self.rounds, data[int(layers[1][0])], color = self.colour[int(layers[1][0])], alpha = 0.3)
      # ax.fill_between(self.rounds, data[int(layers[0][0])], color = self.colour[int(layers[0][0])], alpha = 0.3)
      labels = [l for l in self.labels]
      labels.append("Mean Fairness, $F_T$")
      ax.legend(labels, fontsize = 8)
      ax.set_xlabel("Round", fontsize = 12)
      ax.set_xticks([4,9,14,19,24,29], [5,10,15,20,25,30])
      ax.set_ylabel("Normalised Fairness", fontsize = 12)
      ax.set_title(self.plot_string, fontsize = 12)
      ax.set_ylim([0,1.1])
      ax.set_xlim([min(self.rounds), max(self.rounds)])
      plt.gcf().savefig(path + '_tS_v1.png', dpi = 400)
      return

class UserInputError(Exception):
    pass

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise UserInputError("Please enter correct number of command line arguments")
    if sys.argv[1] not in ["FedAvg", "Ditto", "q_FedAvg", "FedMinMax"]:
        raise UserInputError("Invalid Strategy Provided")
    if sys.argv[2] == "NSLKDD":
        n = 100
        e = 5
    elif sys.argv[2] == "FEMNIST":
        n = 205
        e = 5
    elif sys.argv[2] == "10CIFAR":
        n = 10
        e = 10
        sys.argv[2] = "CIFAR"
    elif sys.argv[2] == "100CIFAR":
        n = 100
        e = 10
        sys.argv[2] = "CIFAR"
    else:
        raise UserInputError("Invalid Dataset Provided")
    if sys.argv[3] not in ["niid", "iid"]:
        raise UserInputError("Invalid Data Setting Provided")
    main(strategy = sys.argv[1], dataset = sys.argv[2], setting = sys.argv[3], clients = n, epochs=e)
