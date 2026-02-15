import warnings
from typing import Optional

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from ucimlrepo import fetch_ucirepo
import os

DATASETS = [
    "iris",
    "wine",
    "breast_cancer",
]

def layer_init_filter(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    
    
def layer_init(layer, w_scale=1.0):
    nn.init.xavier_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

    
def plot_grad_flow(layers, ave_grads, max_grads):
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def moving_average(data, window=100):
    """Smoothing function for plot metrics"""
    data = np.array(data)
    cumsum = np.cumsum(data)
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        if i < window:
            count = i + 1
            smoothed[i] = cumsum[i] / count
        else:
            smoothed[i] = (cumsum[i] - cumsum[i-window]) / window
    return smoothed

def create_datasets_and_loaders(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=2,
    n_classes=5,
    random_state=None,
    batch_size=96,
):

    # Create a sklearn classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
    )
    num_train = int(0.6 * n_samples)
    num_val = int(0.2 * n_samples)

    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = X[num_train : num_train + num_val], y[num_train : num_train + num_val]
    X_test, y_test = X[num_train + num_val :], y[num_train + num_val :]

    # Convert target to one-hot encoding
    y_train = np.eye(n_classes)[y_train]
    y_val = np.eye(n_classes)[y_val]
    y_test = np.eye(n_classes)[y_test]

    # Create a torch dataset
    class ClassificationDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    test_dataset = ClassificationDataset(X_test, y_test)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_classification_data(
    n_samples: int = 100,
    n_features: int = 8,
    n_classes: int = 2,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a classification dataset with train, validation, and test splits.

    Parameters:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of features in the dataset.
        n_classes (int): Number of classes in the dataset.
        val_size (float): Proportion of samples in the validation set.
        test_size (float): Proportion of samples in the test set.
        random_state (Optional[int]): Seed for the random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            The train, validation, and test sets for the features and the targets.
    """
    # Determine the number of clusters per class
    n_clusters_per_class = 2  # Default value; adjust as needed

    # Calculate the minimum number of informative features required
    min_informative = int(np.ceil(np.log2(n_classes * n_clusters_per_class)))

    # Ensure n_informative does not exceed n_features
    n_informative = min(min_informative, n_features)

    # Generate the classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state,
    )

    # Calculate the number of validation and test samples
    n_val = int(val_size * n_samples)
    n_test = int(test_size * n_samples)

    # Split the dataset into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(n_val + n_test), random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=n_test, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 1, shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader from the features and targets.

    Parameters:
        X (np.ndarray): Features.
        y (np.ndarray): Targets.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_classification_data_from_ds(
    dataset_name: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Load a dataset from OpenML (UCI repository) and preprocess it:
    - Detect numerical and categorical features
    - Impute missing values
    - One-hot encode categorical features
    - Standardize all features
    - Encode target labels
    - Split into train/validation/test sets

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays
    """
    # Check if the dataset is in the list of available datasets
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not available. Please choose from {DATASETS}."
        )

    # Check if the dataset file exists
    dataset_path = os.path.join("..", "data", f"{dataset_name}.npz")
    if os.path.exists(dataset_path):
        with np.load(dataset_path, allow_pickle=False) as data:
            X_train = data["X_train"]
            X_val = data["X_val"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_val = data["y_val"]
            y_test = data["y_test"]
        return X_train, X_val, X_test, y_train, y_val, y_test

    # fetch dataset
    ids = {
        "iris": 53,
        "wine": 109,
        "breast_cancer": 17,
    }
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not available. Please choose from {DATASETS}."
        )

    ds = fetch_ucirepo(id=ids[dataset_name])
    X = ds.data.features
    y = ds.data.targets

    # Identify feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Pipelines for numerical and categorical features
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    # Adjust validation size relative to the remaining data
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_temp,
    )

    # Fit the preprocessor on the training data and transform all sets
    if dataset_name in ids:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
    else:
        X_train_processed = X_train
        X_val_processed = X_val
        X_test_processed = X_test

    # Save the dataset as a numpy array
    os.makedirs(os.path.join("..", "data"), exist_ok=True)
    np.savez_compressed(
        os.path.join("..", "data", f"{dataset_name}.npz"),
        X_train=X_train_processed,
        X_val=X_val_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        allow_pickle=False,
    )

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
