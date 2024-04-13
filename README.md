# Federated Fairness Analytics

![GitHub License](https://img.shields.io/github/license/oscardilley/federated-fairness?logoColor=green)
[![Email Badge](https://img.shields.io/badge/Contact-Email-pink)](mailto:tundra.01pitches@icloud.com)
![Static Badge](https://img.shields.io/badge/federated_analytics-red)
![Static Badge](https://img.shields.io/badge/fairness-blue)
![Static Badge](https://img.shields.io/badge/XAI-yellow)

## About

For federated learning to be trusted for real world application, particularly where avoiding bias is of critical importance and where data protection legislation such as GDPR limited the ability of the server to validate decisions against datasets directly - the implications of federated learning upon fairness must be better understood. In the current state of the art surveys, the lack of clear definitions or metrics to quantify fairness in federated learning is cited as a key open problem. This project proposes definitions for a number of notions of fairness with corresponding metrics in order to quanity such fairess. The metrics are used to benchmark a number of existing approaches offering a unique insight into fairness performance and improving explainability and transparency of systems, without violating data privacy.

> **NOTE:** This repository demonstrates the output output of my final year Electrical and Electronic Engineering MENg, Individual Research Project at the University of Bristol.


~~~
Copyable code block
~~~



use a note to mention that it is part of a university masters project

## Tools and Resources

The project would not be possible without the fantastic array of open-source tools and datasets that fasciliate federated learning research. This project utilises:
* [Flower](https://flower.ai/docs/framework/index.html#)
* Hugging Face Datasets including [NSL-KDD](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD) and [CIFAR-10](https://huggingface.co/datasets/cifar10)
* PyTorch
* TensorFlow Federated Datasets to utilise the natural partioning of the LEAF dataset, [FEMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist)

The project uses the Flower simulations and was run on an NVIDIA A40 GPU.


## Notions of Fairness


## Experimental Details


## How to Use


## Assumptions and Applicability

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


