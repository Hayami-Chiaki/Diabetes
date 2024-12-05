# Project Introduction

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
