# README

## Overview

This project demonstrates the use of Optuna for hyperparameter optimization of a Gradient Boosting Classifier in a machine learning context. The classifier is trained on a subset of a dataset, and the hyperparameters are tuned to maximize the F1 score (macro average) using cross-validation. After finding the best hyperparameters, the model is evaluated on a test set, and various metrics are reported.

## Prerequisites

To run this project, you will need the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `optuna`

You can install these libraries using pip:

```bash
pip install numpy pandas scikit-learn optuna
```

## Data

The dataset used in this project is expected to be in a CSV file named `data.csv` and should be located in the `D:\download\` directory. The dataset should have a column named `target` which contains the labels, and the remaining columns should be features.

## Project Structure

The project consists of a single Python script that performs the following steps:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded using `pandas`.
   - A subset of the data is selected randomly to reduce computational requirements.
   - The features and target variable are separated.

2. **Train-Test Split**:
   - The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

3. **Hyperparameter Optimization**:
   - An objective function (`optuna_objective`) is defined to optimize the hyperparameters of a Gradient Boosting Classifier using Optuna.
   - The hyperparameters include `max_depth`, `learning_rate`, `n_estimators`, and `subsample`.
   - Cross-validation is used to evaluate the model, and the F1 score (macro average) is used as the scoring metric.
   - The `optimizer_optuna` function runs the optimization process and returns the best hyperparameters and their corresponding score.

4. **Model Training and Evaluation**:
   - A Gradient Boosting Classifier is instantiated with the best hyperparameters found by Optuna.
   - The model is trained on the training set.
   - Predictions are made on the test set.
   - Various metrics are calculated and printed, including the confusion matrix, accuracy, F1 score (macro average), and classification report.

## Running the Project

To run the project, simply execute the Python script. The script will output the confusion matrix, accuracy, F1 score, and classification report for the test set.

## Example Output

Here is an example of what the output might look like:

```
Confusion Matrix:
[[XX XX]
 [XX XX]]

Accuracy:
0.XX

F1 Score (macro):
0.XX

Classification Report:
              precision    recall  f1-score   support

       class0       0.XX      0.XX      0.XX        XX
       class1       0.XX      0.XX      0.XX        XX

    accuracy                           0.XX       XXX
   macro avg       0.XX      0.XX      0.XX       XXX
weighted avg       0.XX      0.XX      0.XX       XXX
```

Note that the actual values in the output will depend on the dataset and the specific run of the script.

## Conclusion

This project demonstrates the use of Optuna for hyperparameter optimization in a machine learning context. By tuning the hyperparameters of a Gradient Boosting Classifier, we can potentially improve its performance on a given dataset. The script provided serves as a starting point and can be modified to suit different datasets and requirements.