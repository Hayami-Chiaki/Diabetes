import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data=pd.read_csv("D:/cxdownload/data.csv")
y=data['target']
x=data.drop(columns=['id', 'target'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNN()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1 Score: {f1}")
print("分类报告:")
print(classification_report(y_test, y_pred, zero_division=0))
