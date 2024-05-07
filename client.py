import argparse
import mlflow
import warnings
from collections import OrderedDict
import numpy as np
from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import os
import cords_semantics.tags as cords_tags

# Import the prepare_dataset function
from dataset import prepare_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define MLflow experiment name
experiment_name = "Federated-Learning-Energy-Prediction"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = os.path.join("logs", experiment_name, run_name)
mlflow.set_tracking_uri('http://localhost:5000')

class ElectricityModel(nn.Module):
    """Custom model for electricity consumption prediction."""

    def __init__(self, input_dim):
        super(ElectricityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)  # Output layer for regression

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.Sigmoid()(x)


def train(model, trainloader, epochs, criterion, optimizer):
    """Train the model on the training set."""
    model.train()
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            data, target = batch
            optimizer.zero_grad()
            output = model(data.to(DEVICE))
            loss = criterion(output, target.to(DEVICE))
            loss.backward()
            optimizer.step()


def evaluate(model, testloader, criterion):
    """Evaluate the model on the test set."""
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            data, target = batch
            output = model(data.to(DEVICE))
            loss += criterion(output, target.to(DEVICE)).item()
            #accuracy = model.score(data, target)
    loss /= len(testloader.dataset)
    return loss


# Load data using the prepare_dataset function
#num_clients = 10  # Number of clients
#batch_size = 32


# Get client id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--client-id",
    choices=[0, 1],
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)
client_id = parser.parse_known_args()[0].client_id

partition_id = np.random.choice(5)

train_loaders, test_loaders = prepare_dataset(client_id, partition_id)

# Define Flower client
class FlowerElectricityClient(NumPyClient):
    def __init__(self, current_round):
        super().__init__()
        self.model = ElectricityModel(input_dim=5).to(DEVICE)
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.current_round = current_round

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        #round_num = config.get("num_rounds", 0)
        self.set_parameters(parameters)
        with mlflow.start_run(run_name=run_name) as mlflow_run:
            # #Log parameters
            # for k, v in config.items():
            #     mlflow.log_param(k,v)
            #server_round = config["server_round"]
            #Train model
            train(self.model, train_loaders, epochs=5, criterion=self.criterion, optimizer=self.optimizer)
            #Log metrics
            loss = evaluate(self.model, test_loaders, self.criterion)
            mlflow.set_tag(cords_tags.CORDS_RUN, mlflow_run.info.run_id)
            mlflow.set_tag(cords_tags.CORDS_RUN_EXECUTES, "ANN")
            mlflow.set_tag(cords_tags.CORDS_IMPLEMENTATION, "python")
            mlflow.set_tag(cords_tags.CORDS_SOFTWARE, "pytorch")
            mlflow.set_tag(cords_tags.CORDS_MODEL_HASHYPERPARAMETER, self.criterion)
            mlflow.set_tag(cords_tags.CORDS_MODEL_HASHYPERPARAMETER, self.optimizer)
            mlflow.log_metric("loss", loss)
            mlflow.pytorch.log_model(self.model, f"Round_{self.current_round}_Client_{client_id}")

        self.current_round += 1
        return self.get_parameters(config={}), len(train_loaders.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate(self.model, test_loaders, self.criterion)
        return loss, len(test_loaders.dataset), {"loss": loss}


def client_fn(cid: str, current_round):
    """Create and return an instance of Flower `Client`."""
    return FlowerElectricityClient(current_round).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    current_round = 0 
    client = FlowerElectricityClient(current_round=current_round).to_client()
    start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )
