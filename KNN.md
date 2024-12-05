# README for SVM Classification Script

## Overview

This README accompanies a Python script designed to classify data using the Support Vector Machine (SVM) algorithm from the scikit-learn library. The script reads a dataset from a CSV file, preprocesses it, trains an SVM model, makes predictions, and evaluates the model's performance.

## Prerequisites

Before running the script, ensure you have the following:

- Python 3.x installed on your system.
- The necessary Python libraries: pandas, scikit-learn, seaborn, and matplotlib. You can install these libraries using pip:
  ```bash
  pip install pandas scikit-learn seaborn matplotlib
  ```
- A CSV file named `data.csv` containing your dataset. The file should have feature values in all columns except the last one, which should contain the target variable (labels).

## Script Workflow

The script performs the following steps:

1. **Data Loading**:
   - Reads the `data.csv` file using pandas and assigns the feature columns to `X` and the target column to `y`.

2. **Data Splitting**:
   - Splits the dataset into training and testing sets using `train_test_split` from scikit-learn, with 80% of the data dedicated to training and 20% to testing.

3. **Feature Scaling**:
   - Standardizes the feature values using `StandardScaler` to ensure equal contribution from all features during the SVM algorithm's distance calculations.

4. **Model Creation and Training**:
   - Creates an SVM model with an Radial Basis Function (RBF) kernel and trains it on the training set.

5. **Prediction and Evaluation**:
   - Generates predictions on the test set using the trained model.
   - Calculates and prints the model's accuracy.
   - Prints a detailed classification report showcasing precision, recall, F1-score, and support for each class.
   - Computes and displays a confusion matrix as a heatmap using seaborn's `heatmap` function.

## Running the Script

To run the script, follow these steps:

1. Ensure that the `data.csv` file is in the same directory as the script or update the file path in the script accordingly.
2. Execute the script using Python:
   ```bash
   python script_name.py  # Replace 'script_name.py' with the actual name of your script file.
   ```

## Output

The script produces the following output:

- Prints the SVM model's accuracy.
- Prints a detailed classification report that includes precision, recall, F1-score, and support for each class.
- Displays a confusion matrix as a heatmap, visually representing the number of true positives, true negatives, false positives, and false negatives.

## Notes

- You can try different SVM kernels (e.g., 'linear', 'poly') by modifying the `kernel` parameter in the `SVC` function call.
- The `zero_division` parameter in the `classification_report` function is set to 0 to avoid division by zero errors when counts for true positives, true negatives, false positives, or false negatives are zero for a specific class.
- The confusion matrix heatmap uses the 'Blues' colormap and annotates each cell with the corresponding count.

By following these steps, you can classify your data using an SVM model and comprehensively evaluate its performance. Enjoy using the script!