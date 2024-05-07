import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
import torch

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int)-> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

def prepare_dataset(client_id, partition_id):
    # Load the CSV data
    if(client_id==0):
        df = pd.read_csv("trainingDataClient0.csv")
    if(client_id==1):
        df = pd.read_csv("trainingDataClient1.csv")

    
    X = df.drop("Value", axis=1).values
    
    X_scaled = torch.from_numpy(StandardScaler().fit_transform(X))
    y = torch.from_numpy(df["Value"].values).view(-1,1)

    #partition the data into 5 partitions
    (X_partitioned, y_partitioned) = partition(X_scaled, y, 5)[partition_id]

    # Separate features and target
    #X = torch.from_numpy(df.drop("Value", axis=1).values)
    

    # # Convert X to a pandas DataFrame
    # X_df = pd.DataFrame(X, columns=X.columns)

    # # Standardize numerical features
    # numeric_features = X_df.select_dtypes(exclude="object").columns
    # numeric_transformer = StandardScaler()

    # # One-hot encode categorical features if any (not applicable for this dataset)
    # categorical_features = X_df.select_dtypes(include="object").columns
    # categorical_transformer = OneHotEncoder()

    # # Combine transformers
    # preprocessor = ColumnTransformer(
    # [
    #     ("OneHotEncoder", categorical_transformer, categorical_features),
    #      ("StandardScaler", numeric_transformer, numeric_features),        
    # ]
    # )
    # Fit and transform the data
    #X_processed = preprocessor.fit_transform(df)

    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_partitioned, y_partitioned, test_size=0.2, random_state=42
    )

   # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # # Divide data into clients
    # train_loaders = []
    # val_loaders = []
    # clients = np.random.choice(num_clients, len(X_train), replace=True)
    # for client_idx in range(num_clients):
    #     train_mask = clients == client_idx
    #     val_mask = np.logical_not(train_mask)
    #     train_subset = TensorDataset(
    #         torch.tensor(X_train[train_mask], dtype=torch.float32),
    #         torch.tensor(y_train[train_mask], dtype=torch.float32),
    #     )
    #     val_subset = TensorDataset(
    #         torch.tensor(X_train[val_mask], dtype=torch.float32),
    #         torch.tensor(y_train[val_mask], dtype=torch.float32),
    #     )
    #     train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
    #     val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))

    return train_loader, test_loader
