from config import paths
from data_models.data_validator import validate_data
from logger import get_logger, log_error
from prediction.predictor_model import (
    save_predictor_model,
    train_predictor_model,
    set_decision_threshold,
)
from preprocessing.preprocess import (
    insert_nulls_in_nullable_features,
    save_pipeline_and_target_encoder,
    train_pipeline_and_target_encoder,
    transform_data,
    handle_class_imbalance,
)
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, read_json_as_dict, set_seeds, split_train_val
from imbalanced import tune_k_for_smote, tune_class_weights, tune_decision_threshold

logger = get_logger(task_name=__file__)


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir: str = paths.TRAIN_DIR,
    preprocessing_config_file_path: str = paths.PREPROCESSING_CONFIG_FILE_PATH,
    preprocessing_dir_path: str = paths.PREPROCESSING_DIR_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH,
    hpt_results_dir_path: str = paths.HPT_OUTPUTS_DIR,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model
            configuration file.
        train_dir (str, optional): The directory path of the train data.
        preprocessing_config_file_path (str, optional): The path of the preprocessing
            configuration file.
        preprocessing_dir_path (str, optional): The dir path where to save the pipeline
            and target encoder.
        predictor_dir_path (str, optional): Dir path where to save the
            predictor model.
        default_hyperparameters_file_path (str, optional): The path of the default
            hyperparameters file.
        run_tuning (bool, optional): Whether to run hyperparameter tuning.
            Default is False.
        hpt_specs_file_path (str, optional): The path of the configuration file for
            hyperparameter tuning.
        hpt_results_dir_path (str, optional): Dir path where to save the HPT results.
        explainer_config_file_path (str, optional): The path of the explainer
            configuration file.
        explainer_dir_path (str, optional): Dir path where to save the explainer.
    Returns:
        None
    """

    try:

        logger.info("Starting training...")
        # load and save schema
        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        # load model config
        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        # set seeds
        logger.info("Setting seeds...")
        set_seeds(seed_value=model_config["seed_value"])

        # load train data
        logger.info("Loading train data...")
        train_data = read_csv_in_directory(file_dir_path=train_dir)

        # validate the data
        logger.info("Validating train data...")
        validated_data = validate_data(
            data=train_data, data_schema=data_schema, is_train=True
        )

        logger.info("Loading preprocessing config...")
        preprocessing_config = read_json_as_dict(preprocessing_config_file_path)

        # Scenario: one of "baseline", "smote", "class_weights", "decision_threshold"
        scenario = model_config["scenario"]

        logger.info(f"Training scenario: {scenario}")

        # split train data into training and validation sets
        logger.info("Performing train/validation split...")
        if scenario != "baseline":
            train_split, val_split = split_train_val(
                validated_data, val_pct=model_config["validation_split"]
            )
        else:
            # no need to do train/valid split for baseline, nothing to tune
            train_split = validated_data
            val_split = None

        # insert nulls in nullable features if no nulls exist in train data
        logger.info("Inserting nulls in nullable features if not present...")
        train_split_with_nulls = insert_nulls_in_nullable_features(
            train_split, data_schema, preprocessing_config
        )

        # fit and transform using pipeline and target encoder, then save them
        logger.info("Training preprocessing pipeline and label encoder...")
        pipeline, target_encoder = train_pipeline_and_target_encoder(
            data_schema, train_split_with_nulls, preprocessing_config
        )
        transformed_train_inputs, transformed_train_targets = transform_data(
            pipeline, target_encoder, train_split_with_nulls
        )
        if val_split is not None:
            transformed_val_inputs, transformed_val_labels = transform_data(
                pipeline, target_encoder, val_split
            )

        logger.info("Saving pipeline and label encoder...")
        save_pipeline_and_target_encoder(
            pipeline, target_encoder, preprocessing_dir_path
        )

        # Read default hyperparameters
        hyperparameters = read_json_as_dict(default_hyperparameters_file_path)

        # If scenario is smote, tune k for smote and handle class imbalance
        if scenario == "smote":
            logger.info("Tuning K for Smote...")
            best_k = tune_k_for_smote(
                transformed_train_inputs,
                transformed_train_targets,
                transformed_val_inputs,
                transformed_val_labels,
                hyperparameters,
                hpt_specs_file_path,
                hpt_results_dir_path,
            )
            # apply smote to training data
            transformed_train_inputs, transformed_train_targets = (
                handle_class_imbalance(
                    transformed_train_inputs,
                    transformed_train_targets,
                    k_neighbors=best_k,
                )
            )
        elif scenario == "class_weights":
            logger.info("Tuning class weights...")
            best_positive_class_weight = tune_class_weights(
                transformed_train_inputs,
                transformed_train_targets,
                transformed_val_inputs,
                transformed_val_labels,
                hyperparameters,
                hpt_specs_file_path,
                hpt_results_dir_path,
            )
            # update hyperparameters with best positive class weight
            hyperparameters.update(
                {"positive_class_weight": best_positive_class_weight}
            )
        elif scenario == "decision_threshold":
            logger.info("Tuning decision threshold...")
            best_threshold, predictor = tune_decision_threshold(
                transformed_train_inputs,
                transformed_train_targets,
                transformed_val_inputs,
                transformed_val_labels,
                hyperparameters,
                hpt_specs_file_path,
                hpt_results_dir_path,
            )
            # set decision threshold in the model
            set_decision_threshold(predictor, best_threshold)
        elif scenario == "baseline":
            logger.info("No class imbalance handling needed for baseline scenario.")
        else:
            raise ValueError(f"Invalid scenario: {scenario}")

        # train model
        if scenario != "decision_threshold":
            logger.info("Training classifier...")
            predictor = train_predictor_model(
                transformed_train_inputs,
                transformed_train_targets,
                hyperparameters,
            )

        # save predictor model
        logger.info("Saving classifier...")
        save_predictor_model(predictor, predictor_dir_path)

        logger.info("Training completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
