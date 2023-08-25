import math
import os
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss

from config import paths
from prediction.predictor_model import evaluate_predictor_model, train_predictor_model
from utils import read_json_as_dict, save_dataframe_as_csv

HPT_RESULTS_FILE_NAME = "HPT_results.csv"


class HyperParameterTuner:
    """Hyperopt hyperparameter tuner class.

    Args:
        default_hps (Dict[str, Any]): Dictionary of default hyperparameter values.
        hpt_specs (Dict[str, Any]): Dictionary of hyperparameter tuning specs.
        hpt_results_dir_path (str): Dir path to save the hyperparameter tuning
            results.
        is_minimize (bool, optional): Whether the metric should be minimized.
            Defaults to True.
    """

    def __init__(
        self,
        default_hyperparameters: Dict[str, Any],
        hpt_specs: Dict[str, Any],
        hpt_results_dir_path: str,
        is_minimize: bool = True,
    ):
        """Initializes an instance of the hyperparameter tuner.

        Args:
            default_hyperparameters: Dictionary of default hyperparameter values.
            hpt_specs: Dictionary of hyperparameter tuning specs.
            hpt_results_dir_path: Dir path to save the hyperparameter tuning results.
            is_minimize:  Whether the metric should be minimized or maximized.
                Defaults to True.
        """
        self.default_hyperparameters = default_hyperparameters
        self.hpt_specs = hpt_specs
        self.hpt_results_dir_path = hpt_results_dir_path
        self.is_minimize = is_minimize
        self.num_trials = hpt_specs.get("num_trials", 20)
        assert self.num_trials >= 2, "Hyperparameter Tuning needs at least 2 trials"
        # Trials captures the search information
        self.trials = Trials()
        self.hpt_space = self._get_hpt_space()

    def _get_objective_func(
        self,
        train_X: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, np.ndarray],
        valid_X: Union[pd.DataFrame, np.ndarray],
        valid_y: Union[pd.Series, np.ndarray],
    ) -> Callable:
        """Gets the objective function for hyperparameter tuning.

        Args:
            train_X: Training data features.
            train_y: Training data labels.
            valid_X: Validation data features.
            valid_y: Validation data labels.

        Returns:
            A callable objective function for hyperparameter tuning.
        """

        def objective_func(hyperparameters):
            """Build a model from this hyper parameter permutation and evaluate
            its performance"""
            # train model
            classifier = train_predictor_model(train_X, train_y, hyperparameters)
            # evaluate the model
            score = round(evaluate_predictor_model(classifier, valid_X, valid_y), 6)
            if np.isnan(score) or math.isinf(score):
                # sometimes loss becomes inf/na, so use a large "bad" value
                score = 1.0e6 if self.is_minimize else -1.0e6
            # If this is a maximization metric then return negative of it
            return score if self.is_minimize else -score

        return objective_func

    def _get_hpt_space(self) -> Dict[str, Any]:
        """Get the hyperparameter tuning search space.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameter space objects.
        """
        param_grid = {}
        space_map = {
            ("categorical", None): hp.choice,
            ("int", "uniform"): hp.quniform,
            ("int", "log-uniform"): hp.qloguniform,
            ("real", "uniform"): hp.uniform,
            ("real", "log-uniform"): hp.loguniform,
        }
        for hp_obj in self.hpt_specs["hyperparameters"]:
            hp_val_type = hp_obj["type"]
            search_type = hp_obj.get("search_type")
            key = (hp_val_type, search_type)

            if key in space_map:
                func = space_map[key]
                name = hp_obj["name"]
                if hp_val_type == "categorical":
                    val = func(name, hp_obj["categories"])
                else:
                    low = hp_obj["range_low"]
                    high = hp_obj["range_high"]
                    if hp_val_type == "real" and search_type == "log-uniform":
                        # take logarithm of bounds for log-uniform distribution
                        low, high = np.log(low), np.log(high)
                    val = func(name, low, high, 1) \
                        if hp_val_type == "int" else func(name, low, high)
            else:
                raise ValueError(
                    f"Error creating Hyper-Param Grid. "
                    f"Undefined value type: {hp_val_type} "
                    f"or search_type: {search_type}. "
                    "Verify hpt_config.json file."
                )
            param_grid[name] = val

        return param_grid


    def run_hyperparameter_tuning(
        self,
        train_X: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, np.ndarray],
        valid_X: Union[pd.DataFrame, np.ndarray],
        valid_y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Runs the hyperparameter tuning process.

        Args:
            train_X: Training data features.
            train_y: Training data labels.
            valid_X: Validation data features.
            valid_y: Validation data labels.

        Returns:
            A dictionary containing the best model name, hyperparameters, and score.
        """
        objective_func = self._get_objective_func(train_X, train_y, valid_X, valid_y)
        best_hyperparams = fmin(
            # the objective function to minimize
            fn=objective_func,
            # the hyperparameter space
            space=self.hpt_space,
            # search algorithm; This object, such as `hyperopt.rand.suggest` and
            # `hyperopt.tpe.suggest` provides logic for sequential search of the
            # hyperparameter space.
            algo=tpe.suggest,
            # Allow up to this many function evaluations before returning,
            max_evals=self.num_trials,
            # Set seed for reproducibility
            rstate=np.random.default_rng(0),
            # trials captures the search information (we set this in __init__)
            trials=self.trials,
            # early stop
            early_stop_fn=no_progress_loss(10),
            # 
            return_argmin=False
        )
        self.save_hpt_summary_results()
        return best_hyperparams

    def save_hpt_summary_results(self):
        """Save the hyperparameter tuning results to a file."""
        # save trial results
        hpt_results_df = (
            pd.concat(
                [pd.DataFrame(self.trials.vals), pd.DataFrame(self.trials.results)],
                axis=1,
            )
            .sort_values(by="loss", ascending=False)
            .reset_index(drop=True)
        )
        hpt_results_df.insert(0, "trial_num", 1 + np.arange(hpt_results_df.shape[0]))
        if not os.path.exists(self.hpt_results_dir_path):
            os.makedirs(self.hpt_results_dir_path)
        save_dataframe_as_csv(
            hpt_results_df,
            os.path.join(self.hpt_results_dir_path, HPT_RESULTS_FILE_NAME),
        )


def tune_hyperparameters(
    train_X: Union[pd.DataFrame, np.ndarray],
    train_y: Union[pd.Series, np.ndarray],
    valid_X: Union[pd.DataFrame, np.ndarray],
    valid_y: Union[pd.Series, np.ndarray],
    hpt_results_dir_path: str,
    is_minimize: bool = True,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH,
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Scikit-Optimize (SKO) hyperparameter tuner.

    This function creates an instance of the SKOHyperparameterTuner with the
    provided hyperparameters and tuning specifications, then runs the hyperparameter
    tuning process and returns the best hyperparameters.

    Args:
        train_X (Union[pd.DataFrame, np.ndarray]): Training data features.
        train_y (Union[pd.Series, np.ndarray]): Training data labels.
        valid_X (Union[pd.DataFrame, np.ndarray]): Validation data features.
        valid_y (Union[pd.Series, np.ndarray]): Validation data labels.
        hpt_results_dir_path (str): Dir path to the hyperparameter tuning results file.
        is_minimize (bool, optional): Whether the metric should be minimized.
            Defaults to True.
        default_hyperparameters_file_path (str, optional): Path to the json file with
            default hyperparameter values.
            Defaults to the path defined in the paths.py file.
        hpt_specs_file_path (str, optional): Path to the json file with hyperparameter
            tuning specs.
            Defaults to the path defined in the paths.py file.

    Returns:
        Dict[str, Any]: Dictionary containing the best hyperparameters.
    """
    default_hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    hyperparameter_tuner = HyperParameterTuner(
        default_hyperparameters=default_hyperparameters,
        hpt_specs=hpt_specs,
        hpt_results_dir_path=hpt_results_dir_path,
        is_minimize=is_minimize,
    )
    best_hyperparams = hyperparameter_tuner.run_hyperparameter_tuning(
        train_X, train_y, valid_X, valid_y
    )
    return best_hyperparams
