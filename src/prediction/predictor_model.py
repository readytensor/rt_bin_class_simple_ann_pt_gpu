import os
import warnings
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch as T
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from logger import get_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"

logger = get_logger(task_name="pt_model_training")

device = "cuda:0" if T.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def get_activation(activation: str) -> T.nn.Module:
    """
    Return the activation function based on the input string.

    Args:
        activation (str): Name of the activation function.

    Returns:
        T.nn.Module: The requested activation function.

    Raises:
        Exception: If the activation string does not match any known
                   activation functions.
    """
    if activation == "tanh":
        activation = T.tanh
    elif activation == "relu":
        activation = T.relu
    elif activation == "none":
        activation = T.nn.Identity
    else:
        raise Exception(f"Error: Unrecognized activation type: {activation}")
    return activation


class Net(T.nn.Module):
    def __init__(self, D: int, K: int, activation: str) -> None:
        """
        Initialize the neural network.

        Args:
            D (int): Dimension of input data.
            K (int): Dimension of output data.
            activation (str): Activation function to be used in hidden layers.

        Returns:
            None
        """
        super(Net, self).__init__()
        M1 = max(2, int(D * 1.5))
        M2 = 4
        self.activation = get_activation(activation)
        self.hid1 = T.nn.Linear(D, M1)
        self.hid2 = T.nn.Linear(M1, M2)
        self.oupt = T.nn.Linear(M2, K)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Forward pass through the network.

        Args:
            x (T.Tensor): Input to the network.

        Returns:
            x (T.Tensor): Output of the network.
        """
        x = self.activation(self.hid1(x))
        x = self.activation(self.hid2(x))
        x = self.oupt(x)  # no softmax: CrossEntropyLoss()
        return x

    def get_num_parameters(self) -> int:
        """
        Calculate the total number of parameters in the network.

        Returns:
            pp (int): Total number of parameters in the network.
        """
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class CustomDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the dataset.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): Corresponding labels.

        Returns:
            None
        """
        self.x = np.array(x)
        self.y = np.array(y)

    def __getitem__(self, index: int) -> tuple:
        """
        Get one item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: A tuple containing the item and its corresponding label.
        """
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.x)


def get_loss(
    model: T.nn.Module,
    device: str,
    data_loader: T.utils.data.DataLoader,
    loss_function: T.nn.modules.loss._Loss,
) -> float:
    """
    Calculate the average loss over the dataset.

    Args:
        model (T.nn.Module): The model to calculate the loss on.
        device (str): The device on which the calculations will be performed.
        data_loader (T.utils.data.DataLoader): The data loader providing the data.
        loss_function (T.nn.modules.loss._Loss): The loss function to use.

    Returns:
        float: The average loss.
    """
    model.eval()
    loss_total = 0
    with T.no_grad():
        for data in data_loader:
            inputs = data[0].to(device).float()
            labels = data[1].type(T.LongTensor).to(device)
            output = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(output, labels)
            loss_total += loss.item()
    return loss_total / len(data_loader)


class Classifier:
    """A wrapper class for the ANN Binary classifier in PyTorch."""

    model_name = "Simple_ANN_PyTorch_Binary_Classifier"
    min_samples_for_valid_split = 100

    def __init__(
        self,
        D: Optional[int] = None,
        K: Optional[int] = None,
        lr: float = 1e-3,
        activation: str = "relu",
        **kwargs,
    ) -> None:
        """
        Construct a new binary classifier.

        Args:
            D (int, optional): Size of the input layer. Defaults to None (set in `fit`).
            K (int, optional): Size of the output layer.
                                Defaults to None (set in `fit`).
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
            activation (str, optional): Activation function for hidden layer.
                                Defaults to "relu". Options: ["relu", "tanh", "none"]
        """
        self.D = D
        self.K = K
        self.activation = activation
        self.lr = lr
        self._print_period = 10
        # following are set when fitting to data
        self.net = None
        self.criterion = None
        self.optimizer = None

    def _build_model(self):
        """
        Build and set up the model, loss function, and optimizer.
        """
        self.net = Net(D=self.D, K=self.K, activation=self.activation).to(device)
        self.criterion = T.nn.CrossEntropyLoss()
        # self.optimizer = T.optim.SGD(self.net.parameters(), lr=self.lr)
        self.optimizer = T.optim.Adam(self.net.parameters(), lr=self.lr)

    def fit(
        self,
        train_inputs: pd.DataFrame,
        train_targets: pd.Series,
        batch_size: int = 64,
        epochs: int = 1000,
        verbose: int = 1,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Fit the model to the given training data.

        Args:
            train_inputs (pd.DataFrame): Training inputs.
            train_targets (pd.Series): Training targets.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            epochs (int, optional): Number of epochs to train. Defaults to 1000.
            verbose (int, optional): Whether to print training progress.
                                    Defaults to 1.

        Returns:
            List[Dict[str, Union[int, float]]]: Training losses for each epoch.
        """
        N, self.D = train_inputs.shape
        self.K = len(set(train_targets.values))
        self._build_model()

        if N >= self.min_samples_for_valid_split:
            train_X, valid_X, train_y, valid_y = train_test_split(
                train_inputs.values,
                train_targets.values,
                test_size=0.2,
                random_state=42,
            )
        else:
            train_X, valid_X, train_y, valid_y = train_inputs, None, train_targets, None

        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        if valid_X is not None and valid_y is not None:
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            valid_loader = None
        losses = self._run_training(
            train_loader,
            valid_loader,
            epochs,
            use_early_stopping=True,
            patience=10,
            verbose=verbose,
        )

        return losses

    def _run_training(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        epochs: int,
        use_early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 1,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Run the training loop.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (Optional[DataLoader]): DataLoader for validation data.
            epochs (int): Number of epochs to train.
            use_early_stopping (bool, optional): Whether to use early stopping.
                            Defaults to True.
            patience (int, optional): Number of epochs to wait before stopping
                            training when validation loss doesn't decrease.
                            Defaults to 3.
            verbose (int, optional): Whether to print training progress.
                            Defaults to 1.

        Returns:
            List[Dict[str, Union[int, float]]]: Training losses for each epoch.
        """
        best_loss = 1e7
        losses = []
        for epoch in range(epochs):
            for times, data in enumerate(train_loader):
                inputs, labels = data[0].to(device).float(), data[1].type(
                    T.LongTensor
                ).to(device)
                output = self.net(inputs)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # current_loss = loss.item()
            train_loss = get_loss(self.net, device, train_loader, self.criterion)
            epoch_log = {"epoch": epoch, "train_loss": train_loss}

            if valid_loader is not None:
                val_loss = get_loss(self.net, device, valid_loader, self.criterion)
                epoch_log["val_loss"] = val_loss

            # Show progress
            if verbose == 1:
                if epoch % self._print_period == 0 or epoch == epochs - 1:
                    val_loss_str = (
                        ""
                        if valid_loader is None
                        else f", val_loss: {np.round(val_loss, 5)}"
                    )
                    logger.info(
                        f"Epoch: {epoch+1}/{epochs}"
                        f", loss: {np.round(train_loss, 5)}"
                        f"{val_loss_str}"
                    )

            losses.append(epoch_log)

            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = val_loss
                else:
                    current_loss = train_loss

                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        if verbose == 1:
                            logger.info("Early stopping!")
                        return losses

        return losses

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class probabilities.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        X = T.FloatTensor(X).to(device)
        preds = T.softmax(self.net(X), dim=-1).detach().cpu().numpy()
        return preds

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the given data.

        Args:
            inputs (pd.DataFrame): The input data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        class_probs = self._predict(inputs.values)
        class1_probs = class_probs[:, 1]
        predicted_labels = (class1_probs >= 0.5).astype(int)
        return np.squeeze(predicted_labels)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the given data.

        Args:
            inputs (pd.DataFrame): The input data.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        return self._predict(inputs.values)

    def summary(self):
        """
        Print a summary of the neural network.
        """
        self.net.summary()

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Evaluate the model and return the accuracy.

        Args:
            x_test (pd.DataFrame): Training inputs.
            y_test (pd.Series): Training targets.

        Returns:
            float: Accuracy of the model on test data.

        Raises:
            NotFittedError: If the model is not fitted yet.
        """
        if self.net is not None:
            predicted_classes = self.predict(x_test)
            accuracy = np.sum(predicted_classes == y_test) / len(y_test)
            return accuracy
        else:
            raise NotFittedError("Model is not fitted yet.")

    def save(self, model_path: str):
        """
        Save the model to the specified path.

        Args:
            model_path (str): Path to save the model.

        Raises:
            NotFittedError: If the model is not fitted yet.
        """
        if self.net is None:
            raise NotFittedError("Model is not fitted yet.")
        model_params = {
            "D": self.D,
            "K": self.K,
            "lr": self.lr,
            "activation": self.activation,
        }
        joblib.dump(model_params, os.path.join(model_path, MODEL_PARAMS_FNAME))
        T.save(self.net.state_dict(), os.path.join(model_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_path: str) -> "Classifier":
        """
        Load the model from the specified path.

        Args:
            model_path (str): Path to load the model from.

        Returns:
            Classifier: The loaded model.
        """
        model_params = joblib.load(os.path.join(model_path, MODEL_PARAMS_FNAME))
        classifier = cls(**model_params)
        classifier._build_model()
        classifier.net.load_state_dict(
            T.load(os.path.join(model_path, MODEL_WTS_FNAME))
        )
        return classifier


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, model_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        model_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model.save(model_dir_path)


def load_predictor_model(model_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        model_dir_path (str): Dir path to the saved model.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(model_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)


def save_training_history(history, dir_path):
    """
    Save model training history to a JSON file
    """
    hist_df = pd.DataFrame(history.history)
    hist_json_file = os.path.join(dir_path, HISTORY_FNAME)
    with open(hist_json_file, mode="w") as file_:
        hist_df.to_json(file_)
