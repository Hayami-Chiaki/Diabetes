# Diabetes
Prediction of diabetes

# README for Decision Tree Classifier with and without Randomized Search

## Overview

This project demonstrates the application of a Decision Tree Classifier on a dataset, comparing its performance with and without the use of Randomized Search for hyperparameter tuning. The primary goal is to illustrate the potential improvements in model performance that can be achieved through hyperparameter optimization.

## Prerequisites

- Python 3.x
- Pandas
- Scikit-learn
- SciPy (only required for Randomized Search)

To install these dependencies, you can use pip:

```bash
pip install pandas scikit-learn scipy
```

## Data

The dataset used in this project is stored at `D:\\download\\data.csv`. It should contain a column named `target` which represents the labels to be predicted, and the remaining columns are features used for training the model.

## Code Overview

### Decision Tree Classifier without Randomized Search

1. **Data Loading**: The dataset is loaded using Pandas.
2. **Feature and Target Separation**: Features (`x`) and the target (`y`) are separated from the dataset.
3. **Train-Test Split**: The dataset is split into training and testing sets using `train_test_split` from scikit-learn.
4. **Model Training**: A Decision Tree Classifier is instantiated and trained on the training set.
5. **Predictions and Evaluation**: Predictions are made on the testing set, and the F1 score and classification report are printed to evaluate model performance.

### Decision Tree Classifier with Randomized Search

1. **Data Loading and Preparation**: Similar to the previous section, the dataset is loaded and features/target are separated.
2. **Stratified Train-Test Split**: The dataset is split into training and testing sets using `train_test_split` with stratification to ensure a balanced distribution of classes in both sets.
3. **Hyperparameter Distribution**: A parameter distribution is defined for Randomized Search, specifying ranges for `max_depth`, `min_samples_split`, `min_samples_leaf`, and the `criterion` for splitting nodes.
4. **Randomized Search**: `RandomizedSearchCV` from scikit-learn is used to find the best hyperparameters for the Decision Tree Classifier.
5. **Model Training with Best Parameters**: The best classifier found through Randomized Search is trained on the training set.
6. **Predictions and Evaluation**: Predictions are made on the testing set, and the F1 score and classification report are printed to evaluate the optimized model's performance.

## Results

The output of the code will display the F1 score and classification report for both the Decision Tree Classifier without Randomized Search and the Decision Tree Classifier with Randomized Search. This allows for a direct comparison of model performance with and without hyperparameter tuning.

### Example Output

```plaintext
# Without Randomized Search
F1-Score : 0.46
              precision    recall  f1-score   support

           0       0.86      0.97      0.92        50
           1       0.70      0.55      0.62        50

    accuracy                           0.85       100
   macro avg       0.46      0.39      0.41       100
weighted avg       0.46      0.46      0.81       100

# With Randomized Search
F1-Score : 0.46
              precision    recall  f1-score   support

           0       0.75      0.80      0.78        50
           1       0.69      0.46      0.67        50

    accuracy                           0.70       100
   macro avg       0.72      0.73      0.73       100
weighted avg       0.72      0.70      0.71       100
```

Note: The above output is just an example and may differ depending on the actual dataset and the random state used during the split.

## Conclusion

Gradient Boosting Trees: The accuracy rate is 84%. The predictions for category 1 are low. However, after conducting adjustment experiments, it was found that the f1_score increases somewhat as the data volume increases. The specific reasons for this need further exploration.

Default Parameter Decision Tree: Using default parameters, the accuracy rates for categories 1 and 2 are relatively low, but there is some prediction capability.

Random Search Optimized Decision Tree: After using random search to adjust the parameters, the accuracy rate increased, but the predictions for categories 1 and 2 became worse than before. This is likely due to the imbalanced distribution of the sample dataset.

# README for GBDT

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

# README for SVM

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

# README for XGBoost

## Project Overview

This project aims to classify and predict diabetes outcomes using the XGBoost model. The data is sourced from a CSV file containing multiple features and a target variable. The project process includes data preprocessing, feature engineering, model training, and model evaluation. Through this project, you can learn how to use XGBoost for multiclass classification tasks and assess model performance.

## Environment Dependencies

- Python 3.x
- pandas
- scikit-learn
- xgboost

Ensure that you have installed these libraries. If not, you can install them using pip:

```bash
pip install pandas scikit-learn xgboost
```

## Dataset

The dataset is located at `C:\Users\YourUsername\Desktop\Python Experiments\data.csv`. It contains the following features:

- `HighBP`: High Blood Pressure
- `HighChol`: High Cholesterol
- `CholCheck`: Cholesterol Check
- `BMI`: Body Mass Index
- `Smoker`: Smoking Status
- `Stroke`: Stroke History
- `HeartDiseaseorAttack`: Heart Disease or Attack History
- `PhysActivity`: Physical Activity
- `Fruits`: Fruit Consumption
- `HvyAlcoholConsump`: Heavy Alcohol Consumption
- `AnyHealthcare`: Any Healthcare Received
- `NoDocbcCost`: Doctor Consultation without Cost
- `GenHlth`: General Health
- `MentHlth`: Mental Health
- `PhysHlth`: Physical Health
- `DiffWalk`: Difficulty in Walking
- `Sex`: Gender
- `Age`: Age
- `Education`: Education Level
- `Income`: Income

The target variable is `target`, representing the classification outcome.

## Project Workflow

1. **Data Acquisition**: Read the CSV file using pandas.
2. **Dataset Splitting**: Split the dataset into training and testing sets using train_test_split.
3. **Feature Engineering**: Standardize the features using StandardScaler.
4. **Model Configuration**: Set up the parameters for the XGBoost model, including learning rate, maximum depth, objective function, evaluation metric, and number of classes.
5. **Data Conversion**: Convert the training and testing sets into DMatrix format for XGBoost.
6. **Model Training**: Train the XGBoost model with early stopping criteria.
7. **Model Evaluation**: Predict using the test set and calculate the F1 score and accuracy.

## Code Explanation

- `diabetes = pd.read_csv(r'C:\Users\YourUsername\Desktop\Python Experiments\data.csv')`: Reads the dataset.
- `train_test_split`: Splits the dataset.
- `StandardScaler`: Standardizes the features.
- `xgb.DMatrix`: Converts data into DMatrix format.
- `xgb.train`: Trains the XGBoost model.
- `model.predict`: Makes predictions.
- `f1_score` and `accuracy_score`: Evaluate model performance.

## Results Output

The code will output the predictions, F1 score, and accuracy. The F1 score is calculated using the macro average, which is suitable for multiclass classification tasks.

## Notes

- Ensure the dataset path is correct.
- Feature and target variable names should match those in the code.
- Check data format and library versions if any errors or warnings occur.

## Future Improvements

- Experiment with different XGBoost parameters to optimize model performance.
- Use cross-validation to further assess model stability.
- Try other machine learning algorithms for comparison.

This README should help you understand the project's workflow and code. If you have any questions or suggestions, please feel free to reach out.

# README for LightGBM

## Project Introduction

This project mainly implements the training of a model for a multi-classification task on the diabetes dataset using the LightGBM algorithm and calculates the F1 score of the prediction results. The following is a detailed introduction to the functionality and each part of the code.

## 1. Environment Dependencies
The running of this project depends on the following major Python libraries:
- `pandas`: Used for data reading, processing, and analysis. 
- `lightgbm`: This is the core library used in this project to build and train the model.
- `sklearn`: Specifically, the `model_selection` and `metrics` modules are used.

## 2. Data Preparation
1. **Data Reading**:
    - The diabetes dataset is read from the `data.csv` file in the current directory via the statement `Diabetes = pd.read_csv("./data.csv")`.
2. **Data Splitting**:
    - First, the feature data part is extracted from the read dataset `Diabetes`. The columns from the 1st to the 22nd (excluding the first column) are selected as the feature data via `Diabetes_data = Diabetes.iloc[:, 1:22]`.
    - Then, the `train_test_split` function is used to split the feature data `Diabetes_data` and the target data `Diabetes.target` into a training set (`x_train`, `y_train`) and a test set (`x_test`, `y_test`) according to the proportion of `test_size = 0.2` (i.e., the test set accounts for 20% of the total data), and `random_state = 78` is set to ensure the consistency of the splitting results each time.

## 3. Model Construction and Training
1. **Model Parameter Setting**:
    - A dictionary `params_classifier` is defined to store the parameters of the LightGBM classifier. The specific parameters are as follows:
        - `"boosting_type": "gbdt"`: Specifies the boosting type as Gradient Boosting Decision Tree (GBDT).
        - `"objective": "multiclass"`: Indicates that this is a multi-classification task objective function.
        - `"num_leaves": 19`: Sets the number of leaf nodes of the tree.
        - `"learning_rate": 0.3`: The learning rate is 0.3.
        - `"lambda_l1":0.3` and `"lambda_l2":0.1`: These are the L1 and L2 regularization parameters respectively.
        - `"n_estimators":1000`: Sets the number of base learners to 1000.
        - `"feature_fraction": 0.9`: The proportion of features used in each iteration.
        - `"bagging_fraction": 0.6`: The proportion of data used for training in each iteration.
        - `"bagging_freq": 1`: Data sampling is performed once per iteration.
        - `"verbose": -1`: Controls the output information during the training process. Here, it is set to not output detailed information.
        - `"num_class": 3`: Since it is a multi-classification task, the number of classes is explicitly specified as 3 here.
2. **Model Creation and Training**:
    - The LightGBM classifier instance is created according to the set parameters via `model_classifier = lgb.LGBMClassifier(**params_classifier)`.
    - Then, the model is trained by passing the feature data `x_train` and the target data `y_train` of the training set into the model for fitting via `model_classifier.fit(x_train, y_train)`.

## 4. Model Evaluation
1. **Prediction**:
    - After the model training is completed, the feature data `x_test` of the test set is predicted using the trained model via `y_pred_classifier = model_classifier.predict(x_test)`, and the prediction result `y_pred_classifier` is obtained.
2. **F1 Score Calculation and Output**:
    - Finally, the F1 score of the prediction result is calculated via `f1_classifier = f1_score(y_test, y_pred_classifier,average="macro")`. Here, the macro-average (`average="macro"`) method is used to comprehensively evaluate the performance of the multi-classification task. And the calculated F1 score is output via `print(f1_classifier)` so that the performance of the model on the test set can be intuitively understood.

## 5. Running Instructions
To run this project code, make sure that all the above-mentioned dependent libraries (`pandas`, `lightgbm`, `sklearn`) have been installed. Then place the diabetes dataset file (`data.csv`) in the same directory as the code file. Simply run the code file to obtain the F1 score of the model's prediction results on the test set.
