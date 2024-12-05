import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report

data = pd.read_csv("./data.csv")
x = data.drop('target', axis=1) 
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-Score : {f1}")
print(classification_report(y_test, y_pred))