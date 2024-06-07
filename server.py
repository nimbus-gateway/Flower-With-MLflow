from typing import List, Tuple
import mlflow
import flwr as fl
from mlflow.tracking import MlflowClient
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from datetime import datetime
import os

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply loss of each client by number of examples used
    losses = [num_examples * m["loss"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    total_examples = sum([num_examples for num_examples, _ in metrics])
    return {"loss": sum(losses) / total_examples}

experiment_name = "Federated-Learning-Energy-Prediction"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Define config
config = ServerConfig(num_rounds=10)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server
    with mlflow.start_run(run_name=run_name) as mlflow_run:
        start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
            )
