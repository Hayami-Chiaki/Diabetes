# README

## Overview

This project includes a Python script designed to classify data using the Support Vector Machine (SVM) algorithm from the scikit-learn library. The script reads data from a CSV file, preprocesses it by splitting into training and testing sets, standardizing feature values, training an SVM model, making predictions, and evaluating the model's performance through accuracy, a classification report, and a confusion matrix.

## Prerequisites

- Python 3.x installed on your machine.
- Required libraries: pandas, scikit-learn, seaborn, and matplotlib. You can install these via pip:
  ```bash
  pip install pandas scikit-learn seaborn matplotlib
  ```
- A CSV file named `data.csv` containing the dataset to be classified. This file should have feature values in all columns except the last one, which should contain the target variable (labels).

## Script Workflow

1. **Data Loading**:
   - The script reads the CSV file using pandas and assigns the feature columns to `X` and the target column to `y`.

2. **Data Splitting**:
   - The dataset is split into training and testing sets using `train_test_split` from scikit-learn, with 80% of the data dedicated to training and 20% to testing.

3. **Feature Scaling**:
   - Feature values are standardized using `StandardScaler` to ensure equal contribution from all features during distance calculations in the SVM algorithm.

4. **Model Creation and Training**:
   - An SVM model with an Radial Basis Function (RBF) kernel is instantiated and trained on the training set.

5. **Prediction and Evaluation**:
   - The trained model generates predictions on the test set.
   - Model accuracy is calculated and printed.
   - A classification report, detailing precision, recall, F1-score, and support for each class, is printed.
   - A confusion matrix is computed and displayed as a heatmap using seaborn's heatmap function.

## Running the Script

1. Ensure the `data.csv` file is placed in the correct directory or update the file path in the script.
2. Execute the script using Python:
   ```bash
   python script_name.py  # Replace 'script_name.py' with the actual name of your script file.
   ```

## Output

- The script prints the SVM model's accuracy.
- It also prints a detailed classification report showcasing the model's performance across different classes.
- Finally, it displays a confusion matrix as a heatmap, visually representing the number of true positives, true negatives, false positives, and false negatives.

## Notes

- You can experiment with various SVM kernels (e.g., 'linear', 'poly') by modifying the `kernel` parameter in the `SVC` function call.
- The `zero_division` parameter in the `classification_report` function is set to 0 to prevent division by zero errors in cases where the counts for true positives, true negatives, false positives, or false negatives are zero for a specific class.
- The confusion matrix heatmap uses the 'Blues' colormap and annotates each cell with the corresponding count.

By following these steps, you can classify your data using an SVM model and comprehensively evaluate its performance.