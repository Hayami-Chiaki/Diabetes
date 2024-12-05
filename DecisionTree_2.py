import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,f1_score
from scipy.stats import randint
data = pd.read_csv("./data.csv")
x = data.drop('target', axis=1)
y = data['target']
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
clf = DecisionTreeClassifier(random_state=42)
param_dist = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 100),
    'min_samples_leaf': randint(1, 100),
    'criterion': ['gini', 'entropy']  
}
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1)
random_search.fit(x_train, y_train)
best_clf = random_search.best_estimator_
y_pred = best_clf.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro') 
print(f"F1-Score : {f1}")
print(classification_report(y_test, y_pred))