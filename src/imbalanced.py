import json
import pandas as pd

from prediction.predictor_model import (
    evaluate_predictor_model,
    train_predictor_model,
)
from utils import read_json_as_dict
from logger import get_logger
from preprocessing.preprocess import handle_class_imbalance

logger = get_logger(task_name=__file__)


def tune_k_for_smote(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    valid_X: pd.DataFrame,
    valid_y: pd.Series,
    hyperparameters: dict,
    hpt_specs_file_path: str,
    hpt_results_dir_path: str,
) -> int:
    """
    Tune the nearest neighbors parameter for the SMOTE algorithm.

    Args:
        train_X: train inputs.
        train_y: train labels.
        valid_X: valid inputs.
        valid_y: valid labels.
        hyperparameters: The hyperparameters to use for training the model.
        hpt_specs_file_path: Path to the hyperparameter tuning specifications file.
        hpt_results_dir_path: Directory to save hyperparameter tuning results.

    Returns:
        int: The best k found.
    """
    logger.info("Tuning K for Smote...")
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    grid_search_vals = hpt_specs["k_neighbors"]["grid_search_vals"]
    best_val_score = -1
    best_k = None
    hpt_vals = []
    for k_val in grid_search_vals:

        balanced_train_inputs, balanced_train_targets = handle_class_imbalance(
            train_X.copy(), train_y.copy(), k_neighbors=k_val
        )
        predictor = train_predictor_model(
            balanced_train_inputs,
            balanced_train_targets,
            hyperparameters,
        )
        val_score = evaluate_predictor_model(predictor, valid_X, valid_y)
        logger.info(f"k_val = {k_val}, val_score = {val_score}")
        hpt_vals.append({"k_val": k_val, "val_score": val_score})
        if val_score > best_val_score:
            best_val_score = val_score
            best_k = k_val

    logger.info(f"Best k_val = {best_k}; " f"Validation score = {best_val_score}")

    # save hyperparameter tuning results
    logger.info("Saving hyperparameter tuning results...")
    hpt_results_file_path = f"{hpt_results_dir_path}/smote_k.json"
    with open(hpt_results_file_path, "w", encoding="utf-8") as f_h:
        json.dump(hpt_vals, f_h, indent=4)

    return best_k


def calculate_positive_class_weights(
    imbalance_ratio: float, balancing_factors: list
) -> list:
    """
    Calculate positive class weights based on imbalance ratio and balancing factors.
    Note the negative class always has a weight of 1.0.
    The weights would from 1.0 (no balancing) to imbalance_ratio (full balancing).

    Args:
        imbalance_ratio: The ratio of positive class frequency to negative class frequency.
        balancing_factors: List of balancing factors to scale the positive class weight.

    Returns:
        list: A list of of positive class weights.
    """
    positive_class_weights_list = []

    for factor in balancing_factors:
        # Linearly interpolate the positive class weight between 1.0 and imbalance_ratio
        positive_class_weight = 1.0 + factor * (imbalance_ratio - 1.0)
        positive_class_weights_list.append(positive_class_weight)
    return positive_class_weights_list


def tune_class_weights(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    valid_X: pd.DataFrame,
    valid_y: pd.Series,
    hyperparameters: dict,
    hpt_specs_file_path,
    hpt_results_dir_path,
):
    """
    Tune the class weights for the predictor model.

    Args:
        train_X: train inputs.
        train_y: train labels.
        valid_X: valid inputs.
        valid_y: valid labels.
        hyperparameters: The hyperparameters to use for training the model.
        hpt_specs_file_path: Path to the hyperparameter tuning specifications file.
        hpt_results_dir_path: Directory to save hyperparameter tuning results.

    Returns:
        float: The best decision threshold found.
    """
    logger.info("Tuning class weights...")

    # Determine the class imbalance ratio
    class_counts = train_y.value_counts().tolist()
    if class_counts[0] > class_counts[1]:
        common_class_count = class_counts[0]
        rare_class_count = class_counts[1]
    else:
        common_class_count = class_counts[1]
        rare_class_count = class_counts[0]
    imbalance_ratio = common_class_count / rare_class_count
    logger.info(f"Imbalance ratio = {imbalance_ratio}")

    # Generate grid search values based on balancing factors
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    balancing_factors = hpt_specs["balancing_factors"][
        "grid_search_vals"
    ]  # List like [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    positive_class_weights_list = calculate_positive_class_weights(
        imbalance_ratio, balancing_factors
    )

    best_val_score = -1
    best_positive_class_weight = None
    hpt_vals = []

    for positive_class_weight in positive_class_weights_list:
        hyperparameters["positive_class_weight"] = positive_class_weight

        predictor = train_predictor_model(train_X, train_y, hyperparameters)
        val_score = evaluate_predictor_model(predictor, valid_X, valid_y)

        logger.info(
            f"positive_class_weight = {positive_class_weight}, val_score = {val_score}"
        )
        hpt_vals.append(
            {"positive_class_weight": positive_class_weight, "val_score": val_score}
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_positive_class_weight = positive_class_weight

    logger.info(
        f"Best positive_class_weight = {best_positive_class_weight}; "
        f"Validation val score = {best_val_score}"
    )

    # save hyperparameter tuning results
    logger.info("Saving hyperparameter tuning results...")
    hpt_results_file_path = f"{hpt_results_dir_path}/positive_class_weights.json"
    with open(hpt_results_file_path, "w", encoding="utf-8") as f_h:
        json.dump(hpt_vals, f_h, indent=4)
    return best_positive_class_weight


def tune_decision_threshold(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    valid_X: pd.DataFrame,
    valid_y: pd.Series,
    hyperparameters: dict,
    hpt_specs_file_path,
    hpt_results_dir_path,
):
    """
    Tune the decision threshold for the predictor model.

    Args:
        train_X: train inputs.
        train_y: train labels.
        valid_X: valid inputs.
        valid_y: valid labels.
        hyperparameters: The hyperparameters to use for training the model.
        hpt_specs_file_path: Path to the hyperparameter tuning specifications file.
        hpt_results_dir_path: Directory to save hyperparameter tuning results.

    Returns:
        float: The best decision threshold found.
    """
    logger.info("Tuning decision threshold...")
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    grid_search_vals = hpt_specs["decision_threshold"]["grid_search_vals"]
    best_val_score = -1
    best_threshold = None
    hpt_vals = []

    logger.info("Training classifier...")
    predictor = train_predictor_model(
        train_X,
        train_y,
        hyperparameters,
    )

    for val in grid_search_vals:
        val_score = evaluate_predictor_model(
            predictor,
            valid_X,
            valid_y,
            decision_threshold=val,
        )
        logger.info(f"decision_threshold = {val}, val_score = {val_score}")
        hpt_vals.append({"decision_threshold": val, "val_score": val_score})
        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = val

    logger.info(
        f"Best decision_threshold = {best_threshold}; "
        f"Validation val score = {best_val_score}"
    )

    # save hyperparameter tuning results
    logger.info("Saving hyperparameter tuning results...")
    hpt_results_file_path = f"{hpt_results_dir_path}/decision_threshold.json"
    with open(hpt_results_file_path, "w", encoding="utf-8") as f_h:
        json.dump(hpt_vals, f_h, indent=4)

    return best_threshold, predictor
