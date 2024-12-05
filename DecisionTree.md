# Readme for Decision Tree Classifier with and without Randomized Search

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