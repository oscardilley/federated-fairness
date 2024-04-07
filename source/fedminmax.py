"""
-------------------------------------------------------------------------------------------------------------

fedminmax.py, v1.0 
by Oscar, March 2024

-------------------------------------------------------------------------------------------------------------

Implementation of the Ditto Personalisation Based Federated Learning Strategy.
Abstracted from the FedAvg strategy from Flower.

Implemented from the paper "Minimax Demographic Group Fairness in Federated Learning"
https://doi.org/10.1145/3531146.3533081

-------------------------------------------------------------------------------------------------------------

Declarations, functions and classes:
- FedMinMax - an abstraction of the Flower strategy.
- data_preprocess - a function to determine the necessary parameters for the FedMinMax strategy.

-------------------------------------------------------------------------------------------------------------

Usage:
Import from root directory using:
    >>> from source.fedminmax import FedMinMax
Pass the necessary inputs to the strategy when instantiating, for example:
    >>> 

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
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

def data_preprocess(num_clients, trainloaders, valloaders, sensitive_attributes):
    """
    Used to parse the data sets and determine:
        - the total number of samples per client
        - the total number of samples for each sensitive attribute per client
        - the total number of samples corresponding to each sensitive attributes
        - the total number of samples
    As we are using simulation, this can be precomputed. In a real federated learning scenario, clients would have to compute their own
    metadata and share with the server. This is a limitation of FedMinMax.

    Inputs:
    num_clients - the total number of clients in the federation.
    trainloaders - a list of Pytorch DataLoaders for all clients, indexed by cid
    valloaders - a list of Pytorch validation DataLoaders for all clients, indexed by cid
    senstive_attributes - the list of sensitive attrributes (in this case, certain labels of the dataset)
    NB: it is assumed that the input data has been suitable preprocessed and that the trainloaders and valloaders are the 
    same length, aligned by index.

    Outputs:
    A dictionary containing the total number of samples, total number of sensitive samples per attribute
    and the total number of each sensitive attribute that each clients dataset contrains.

    """
    print("-------------------Preprocessing Data for FedMinMax--------------------------")
    # Initialising data structures:
    num_sensitive = len(sensitive_attributes)
    num_total_samples = 0
    num_total_sensitive_samples = [0 for i in range(num_sensitive)]
    #num_sensitive_per_client = [[0 for i in range(num_sensitive)] for c in range(num_clients)] # not required
    # Iterating over the clients:
    for i in range(num_clients):
        print(f"Preprocessing client {i}")
        # Parsing the individual datasets:
        for data in trainloaders[i]:
            if "label" in data:
                labels = data["label"]
            else:
                labels = data["class"] # accounts for NSL-KDD
            for s in range(num_sensitive):
                matched = int((labels == sensitive_attributes[s]).sum())
                #num_sensitive_per_client[i][s] += matched
                num_total_sensitive_samples[s]+= matched
                num_total_samples += len(labels)
        for data in valloaders[i]:
            if "label" in data:
                labels = data["label"]
            else:
                labels = data["class"] # accounts for NSL-KDD
            for s in range(num_sensitive):
                matched = int((labels == sensitive_attributes[s]).sum())
                #num_sensitive_per_client[i][s] += matched
                num_total_sensitive_samples[s]+= matched
                num_total_samples += len(labels)
    # returning final metrics:
    print("-------------------------------- End --------------------------------")
    return {"total": num_total_samples, "total_sensitive": num_total_sensitive_samples}#, "sensitive_per_client":num_sensitive_per_client}


class FedMinMax(Strategy):
    """
    FedMinMax Federated Learning Strategy adapted from the Flower FedAvg Strategy
    Implementation based on https://doi.org/10.1145/3531146.3533081

    Additional Attributes for FedMinMax:
    

    Modifications to Standard Strategy for FedMinMax:
        
    """
    def __init__(
        self,
        *, # keyword only arguments after
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = False,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        lr = 0.001,
        adverse_lr = 0.001,
        dataset_information = None,
        sensitive_attributes = []
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.lr = lr # client learning rate
        self.adverse_lr =adverse_lr # learning rate of the adversary at the server
        self.sensitive_attributes = sensitive_attributes
        self.total_samples = dataset_information["total"]
        self.total_sensitive = dataset_information["total_sensitive"]
        #total_sensitive_per_client = dataset_information["sensitive_per_client"] # probably not required
        self.rho = np.array([(i / self.total_samples) for i in self.total_sensitive])
        self.mu = self.rho # weighing coefficients initialised to rho
        self.risks = None


    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedMinMax(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        if self.risks is None:
            pass
        else:
            # Reassigning the weighing coefficients (projected gradient ascent to maximise objective)
            # Using Euclidean projection from: https://doi.org/10.1145/1390156.1390191
            cum_risks = [np.sum([a[i] for a in self.risks]) for i in range(len(self.sensitive_attributes))]
            print(self.risks)
            print(cum_risks)
            avg_risks = [(cum_risks[i] / self.total_sensitive[i]) for i in range(len(self.sensitive_attributes))]
            print(avg_risks)

            # computing unbiased estimate of gradient of 
            gradients = avg_risks



            print(gradients)
            self.mu = self.mu + (self.adverse_lr*gradients)
            print(self.mu)
        print(self.mu)
        # calculate the weights:
        weights = (self.mu / self.rho) # weights initialised to 1,1,1,1,...
        # Passing weights, learning rate and sensitive attributes to each client
        fedminmax_parameters = {"opt": "fedminmax",
                            "w": weights,
                            "lr": self.lr, 
                            "attributes": self.sensitive_attributes
                            }
        # Ditto parameters are appended and the presence of the Ditto config dict acts as strategy flag:
        config["fedminmax"] = fedminmax_parameters
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated