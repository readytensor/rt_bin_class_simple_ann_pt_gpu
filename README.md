# Imbalanced Dataset Handling Branch

This branch provides a simplified implementation specifically designed to handle class imbalance scenarios in binary classification tasks. It focuses solely on comparing different techniques for addressing class imbalance, without additional complexities like hyperparameter tuning, explainable AI, or model serving.

## Features

This branch extends the main implementation with the following class imbalance handling techniques:

1. **Baseline**: No special handling (do nothing)
2. **SMOTE**: Synthetic Minority Over-sampling Technique
3. **Class Weights**: Adjusting class weights to penalize misclassification of the minority class
4. **Threshold Adjustment**: Tuning the classification threshold

## Usage

To use this branch, you need to specify the scenario in the `./src/config/model_config.json` file. Set the "scenario" field to one of the following values:

- `baseline`
- `smote`
- `class_weights`
- `decision_threshold`

Example:

```json
{
  "scenario": "smote",
  ...
}
```

Once the scenario is set, the training process will automatically handle the rest, including:

- Data preprocessing
- Applying the specified imbalance handling technique
- Hyperparameter tuning (if applicable)
- Model training with tuned hyperparameters

## Implementation Details

The main logic for handling different scenarios is implemented in the `run_training()` function. This function:

1. Loads and validates the data
2. Performs train/validation split (except for `baseline`)
3. Applies the specified imbalance handling technique
4. Trains the model with appropriate parameters
5. Saves the trained model and related artifacts

## Simplified Structure

This branch has been streamlined to focus exclusively on class imbalance handling. As a result, the following features have been removed:

- Bayesian hyperparameter tuning
- Explainable AI (LIME or Shapley)
- Model serving functionality
- Unit tests

The goal is to provide a clear, focused implementation for comparing different class imbalance handling techniques.

## Note

This branch is a specialized version of the main implementation, focusing solely on class imbalance scenarios. For the full feature set, please refer to the main branch.

For any issues or questions specific to this simplified imbalance handling implementation, please open an issue in this branch.
