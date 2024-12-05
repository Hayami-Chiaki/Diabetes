import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

Diabetes = pd.read_csv("./data.csv")

Diabetes_data = Diabetes.iloc[:, 1:22]

x_train, x_test, y_train, y_test = train_test_split(
    Diabetes_data, Diabetes.target, random_state=78, test_size=0.2
)

params_classifier = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_leaves": 19,
    "learning_rate": 0.3,
    "lambda_l1":0.3,
    "lambda_l2":0.1,
    "n_estimators":1000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
    "verbose": -1,
    "num_class": 3,
}

model_classifier = lgb.LGBMClassifier(**params_classifier)

model_classifier.fit(x_train, y_train)

y_pred_classifier = model_classifier.predict(x_test)

f1_classifier = f1_score(y_test, y_pred_classifier,average="macro")

print("f1_score:",f1_classifier)

print("分类报告:")
print(classification_report(y_test, y_pred_classifier, zero_division=0))
