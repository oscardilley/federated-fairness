# Federated Fairness Analytics

![GitHub License](https://img.shields.io/github/license/oscardilley/federated-fairness?logoColor=green)
[![Email Badge](https://img.shields.io/badge/Contact-Email-pink)](mailto:tundra.01pitches@icloud.com)
![Static Badge](https://img.shields.io/badge/federated_analytics-red)
![Static Badge](https://img.shields.io/badge/fairness-blue)
![Static Badge](https://img.shields.io/badge/XAI-yellow)

## About

For federated learning to be trusted for real world application, particularly where avoiding bias is of critical importance and where data protection legislation such as GDPR limits the ability of the server to validate decisions against datasets directly - the implications of federated learning upon fairness must be better understood. In the current state of the art surveys, the lack of clear definitions or metrics to quantify fairness in federated learning is cited as a key open problem. This project proposes definitions for a number of notions of fairness with corresponding metrics in order to quanity such fairness. The metrics are used to benchmark a number of existing approaches offering a unique insight into fairness performance and improving explainability and transparency of systems, without violating data privacy.

> **NOTE:** This repository demonstrates the output of my final year Electrical and Electronic Engineering MEng, Individual Research Project at the University of Bristol.
> 

## Tools and Resources

The project would not be possible without the fantastic array of open-source tools and datasets that fasciliate federated learning research. This project utilises:
* [Flower](https://flower.ai/docs/framework/index.html#)
* Hugging Face Datasets including [NSL-KDD](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD) and [CIFAR-10](https://huggingface.co/datasets/cifar10)
* PyTorch
* TensorFlow Federated Datasets to utilise the natural partioning of the LEAF dataset, [FEMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist)

The project uses the Flower simulations and was run on an **NVIDIA A40 GPU**.

## Notions of Fairness

> **$J(x)$** represents *Jain's Fairness index*, a common measure of uniformity of $x$ over $N$ datapoints. Where $J(x) = \frac{(\sum x)^2}{N \times \sum x^2}$

General fairness in federated learning is broken into the following four, symptomatic notions $\in (0,1]$:


### 1. **Individual Fairness, $f_j$**
*Do all clients perform proportionately to their contribution?*

$f_i = J(G_{n,k})$ where the gain, $G_{n,k} = \frac{x_{n,k}}{s_{n,k}}$ for objective function outcome (typically classification accuracy), $x_{n,k}$ and federated Shapley contribution, $s_{n,k}$ for client $n$ in round $k$.


### 2. **Protected Group Fairness, $f_g$**
*Do subgroups of the population that exhibit sensitive attributes perform equivalently to those without?*

$f_g = mean(med(\sum|EOD_{n,a,k}|))$ where for each sensitive label, $a$ in set $A$, the absolute value of the equality odds, $EOD$ is measured at each client in each round. The mean of the median values at each client indicates a measure for the protected group performance over the federation.


### 3. **Incentive Fairness, $f_r$**
*Are clients rewarded proportionately to their contributions and in equal timeframes?*

$f_r = J(R_{n,k})$ where the gain, $R_{n,k} = \frac{r_{n,k}}{s_{n,k}}$ for client reward outcome (typically classification accuracy of the model received from the start of the training round on the clients own dataset when using federated evaluation), $r_{n,k}$ and federated Shapley contribution, $s_{n,k}$ for client $n$ in round $k$.


### 4. **Orchestrator Fairness, $f_o$**
*Does the server succeed in its role of orchestrating a learning ecosystem that maximises the objective function?*

$f_o = \frac{1}{N} \sum x_{n,k}$ measures the average performance of the clients in the federation. This can be compared to centralised performance if feasible. 


### **General fairness, $F_T$**
General fairness is proposed as the weighted sum of the above notions, in the case that each notion is weighted equally, the following expression arises to define $F_T$:

$F_T = \frac{(f_j + f_g + f_r + f_o)}{4}$

## Experimental Details
The project benchmarks use the following variables:
* **Approach/ strategy** - [FedAvg](https://doi.org/10.48550/arXiv.1602.05629), [q-FedAvg](https://doi.org/10.48550/arXiv.1905.10497), [Ditto](https://doi.org/10.48550/arXiv.2012.04221) and [FedMinMax](https://doi.org/10.1145/3531146.3533081)
* **Datasets and number of clients** - CIFAR-10 with 10 clients, CIFAR-10 with 100 clients, NSL-KDD with 100 clients, Federated-EMNIST (FEMNIST) with 205 clients.
* **Heterogeneity** - each dataset is simulated with IID and non-IID partioning between clients. The Direchlet partitioner is used to emulate the non-IID setting for NSL-KDD and CIFAR, with values of $\alpha$ varying between datasets. The non-IID case for the FEMNIST dataset is achieved using natural partitioning per writer. 

## How to Use

Clone the repository and install the necessary imports:
~~~bash
pip install -r requirements.txt
~~~

Select an experiment of choice from the root directory, conditions are indicated by filename and the config parameters at the top of the script may be adjusted. Run the experiment from the root, for example using:
~~~bash
python fedavg_cifar_iid_100c_v1.py
~~~

> **NOTE:** for FEMNIST simulations, a .pickle file containing a copy of the appropriately paritioned PyTorch dataloaders must be provided. This is because the dataset used was augmented a TensorFlow Federated Dataset before the release of Flower Datasets. An alternative would be to use the power of Flower and Hugging Face datasets to implement the EMNIST partioning as similarly to what has been achieved for NSL-KDD and CIFAR-10, as in *source.load_cifar.py* for example.

## Sample Results

Using the plotting scripts provided. Results such as below can be obtained:

![imgs](https://github.com/oscardilley/federated-fairness/blob/main/Results/Plots/FedAvg_CIFAR_iid_100C_bar.png)
![imgs](https://github.com/oscardilley/federated-fairness/blob/main/Results/Plots/FedAvg_CIFAR_iid_100C_tS.png)

## Assumptions and Applicability

Federated learning comes in many flavours. In order to constrain the scope, this research is applicable to federated learning systems with the following characteristics:

* **Centrally orchestrated** – there exists a single central entity that organises the learning, is responsible for initiating training rounds, aggregating the models and selecting clients. This is referred to as the server and orchestrator interchangeably and will be assumed as trustworthy. Implementations using distributed or blockchain control are out of scope for this project. 

* **Horizontal** – horizontal federated learning is assumed, where the feature space is consistent across clients as is the case in most federated learning applications including Google’s Gboard, this is for simplicity to focus on fairness without considering the problem of entity alignment in vertical federated learning. 

* **Task Agnostic** – this work does not have a specific application in mind and the optimisation of the model is considered to be out of scope. Configurations that achieve satisfactory performance in centralised settings are selected and deployed to the clients. 

* **Known Sensitive Attributes** – the labels corresponding to protected groups must be known by the clients and server in order to be measured.

* **Blackbox Clients** – no information is available about the clients, for example regarding database size, dataset distribution, intended participation rate, communication capability, processing ability. However, it can be assumed that the client is capable of processing the model in question.


## Repository Structure

~~~bash
├── LICENSE
├── README.md
├── Results
│   ├── Ditto_CIFAR_iid_100C_5PC_10E_30R_v1.json
│   ├── ...
│   │   <.json results files>
│   ├── ...
│   ├── q_FedAvg_NSLKDD_niid_100C_5PC_5E_30R_v3.json
│   ├── Old
│   │   └── archived .json results
│   ├── Plots
│   │   ├── Ditto_CIFAR_iid_100C_bar.png
│   │   ├── ...
│   │   │   <.png result images>
│   │   ├── ...
│   │   └── q_FedAvg_NSLKDD_niid_100C_tS.png
├── ditto_cifar_iid_100c_v1.py
├── ...
├── <individual experiment .py files>
├── ...
├── q_fedavg_nslkdd_niid_100c_v1.py
├── plotter_v1.py
└── source 
    ├── cifar_net.py 
    ├── client.py
    ├── ditto.py
    ├── fedminmax.py
    ├── femnist_net.py
    ├── load_cifar.py
    ├── load_nslkdd.py
    ├── nslkdd_net.py
    └── shapley.py

~~~


## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


    http://www.apache.org/licenses/LICENSE-2.0


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


