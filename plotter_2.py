"""
-------------------------------------------------------------------------------------------------------------

plotter_2.py, , v1.0
by Oscar, April 2024

-------------------------------------------------------------------------------------------------------------

MatPlotLib Plotting Script for creating plots that compare different experiments directly

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

import matplotlib.pyplot as plt
import numpy as np
import json
import sys

def main(file_names, input_args):
    """ Main entry point"""
    NUMBER_REPEATS = 3
    temp = [None for i in range(NUMBER_REPEATS)]
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
    general_fairness = []
    performance = []
    rounds = []
    labels = []
    r = 20 # taking average of round r to end for scatter plot
    for file in file_names:
        for repeat in range(NUMBER_REPEATS):
            with open("./Results/" + file + f"_v{repeat+1}.json", "r") as openfile:
                temp[repeat] = json.load(openfile)
        # Data processing
        data = averager(temp, data_structure)
        average = np.array([np.mean(np.array([data["general_fairness"]["f_j"][i], data["general_fairness"]["f_g"][i], data["general_fairness"]["f_r"][i], data["general_fairness"]["f_o"][i]])) for i in range(len(data["rounds"]))])
        f_o = np.array([data["general_fairness"]["f_o"][i] for i in range(len(data["rounds"]))])
        general_fairness.append(average) # Imported the JSON as a dictionary
        performance.append(f_o)
        rounds.append(np.array(data["rounds"]))
    # Label processing from filename using underscore delimitter:
    datasets = {"CIFAR": "CIFAR-10", "FEMNIST": "FEMNIST", "NSLKDD": "NSL-KDD"}
    clients = {"10C": "10 Clients", "100C": "100 Clients", "205C": "205 Clients"}
    for file in file_names:
        parts = file.split("_")
        if parts[0] == "q":
            parts.remove("q")
            parts[0] = "q-FedAvg"
        labels.append(f"{parts[0]}, {datasets[parts[1]]} {parts[2]}, {clients[parts[3]]}")
    PATH = f'./Results/Plots/{datasets[parts[1]]}{clients[parts[3]]}'
    time_series(rounds, general_fairness, PATH, labels, datasets[parts[1]])
    scatter([np.mean(i[r:]) for i in performance], [np.mean(i[r:]) for i in general_fairness], PATH, labels, datasets[parts[1]])
    return


def time_series(x, y, savepath, labels = [], dataset="XXX"):
    """ Plotting general fairness timeseries of the inputted files """
    markers = ["p", "D", "^", "o", "s", "P"]
    num_lines = len(x)
    fig,ax = plt.subplots(1)
    fig.suptitle(f"Timeseries Fairness Evaluation\n{dataset} Cross-Device", fontsize = 15)
    ax.grid(linewidth = 0.5, linestyle = "--")
    ax.set_axisbelow(True)
    for p in range(num_lines):
        ax.scatter(x[p], y[p], s=15, marker=markers[p])
        # a,b,c,d,e,f = np.polyfit(x[p], y[p], 5)
        # ax.plot(x[p], (a*(x[p]**5)) + (b*(x[p]**4)) + (c*(x[p]**3)) + (d*(x[p]**2)) + e*x[p] + f, linewidth = 1.2, label = labels[p])
        a,b,c,d = np.polyfit(x[p], y[p], 3)
        ax.plot(x[p], (a*(x[p]**3)) + (b*(x[p]**2)) + c*x[p] + d, linewidth = 1.2, label = labels[p])
        # a,b,c = np.polyfit(x[p], y[p], 2)
        # ax.plot(x[p], (a*(x[p]**2)) + b*x[p] + c, linewidth = 1.2, label = labels[p])
    ax.set_xticks([4,9,14,19,24,29], [5,10,15,20,25,30])
    ax.set_ylim([0,1])
    ax.set_xlim([0, 30])
    ax.legend(fontsize = 8)
    ax.set_ylabel("General Fairness, $F_T$", fontsize = 12)
    ax.set_xlabel("Round", fontsize = 12)
    plt.gcf().savefig(savepath + '_tS_v1.png', dpi = 400)
    return
    

def scatter(x , y , savepath, labels = [], dataset = "XXX"):
    """ Showing the performance of a range of implementations"""
    markers = ["p", "D", "^", "o", "s", "P"]
    num_lines = len(x)
    fig,ax = plt.subplots(1)
    fig.suptitle(f"Comparitive Fairness Evaluation\n{dataset} Cross-Device", fontsize = 15)
    ax.grid(linewidth = 0.5, linestyle = "--")
    ax.set_axisbelow(True)
    for p in range(num_lines):
        ax.scatter(x[p], y[p], s = 50, label=labels[p], marker=markers[p])
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.legend(fontsize=8, loc = "lower left")
    ax.set_ylabel("General Fairness, $F_T$", fontsize = 12)
    ax.set_xlabel("Performance, $f_o$", fontsize = 12)
    plt.gcf().savefig(savepath + '_scatter_v1.png', dpi = 400)
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
    
    return average_dicts(data, skeleton)


if __name__ == "__main__":
    # Select the experiments you want to be processed by listing their file name below
    file_names = [
                  "Ditto_NSLKDD_niid_100C_5PC_5E_30R",
                  "Ditto_NSLKDD_iid_100C_5PC_5E_30R",
                  "q_FedAvg_NSLKDD_niid_100C_5PC_5E_30R",
                  "q_FedAvg_NSLKDD_iid_100C_5PC_5E_30R",
                  "FedAvg_NSLKDD_niid_100C_5PC_5E_30R",
                  "FedAvg_NSLKDD_iid_100C_5PC_5E_30R", 
                #   "FedAvg_FEMNIST_niid_205C_2PC_5E_30R",
                #   "FedAvg_FEMNIST_iid_205C_2PC_5E_30R",
                #   "Ditto_FEMNIST_niid_205C_2PC_5E_30R",
                #   "Ditto_FEMNIST_iid_205C_2PC_5E_30R",
                #   "q_FedAvg_FEMNIST_niid_205C_2PC_5E_30R",
                #   "q_FedAvg_FEMNIST_iid_205C_2PC_5E_30R",
                #   "FedAvg_CIFAR_niid_100C_5PC_10E_30R",
                #   "FedAvg_CIFAR_iid_100C_5PC_10E_30R",
                #   "Ditto_CIFAR_niid_100C_5PC_10E_30R",
                #   "Ditto_CIFAR_iid_100C_5PC_10E_30R",
                #   "q_FedAvg_CIFAR_niid_100C_5PC_10E_30R",
                #   "q_FedAvg_CIFAR_iid_100C_5PC_10E_30R",
                #   "FedAvg_CIFAR_niid_10C_50PC_10E_30R",
                #   "FedAvg_CIFAR_iid_10C_50PC_10E_30R",
                #   "Ditto_CIFAR_niid_10C_50PC_10E_30R",
                #   "Ditto_CIFAR_iid_10C_50PC_10E_30R",
                #   "q_FedAvg_CIFAR_niid_10C_50PC_10E_30R",
                #   "q_FedAvg_CIFAR_iid_10C_50PC_10E_30R",

    ]
    main(file_names, sys.argv[1:])

