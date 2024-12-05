import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("./data.csv")

# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

Diabetes_data = data.iloc[0:10000, 1:22]

X_train, X_test, y_train, y_test = train_test_split(
    Diabetes_data, data.target.iloc[0:10000], random_state=78, test_size=0.2
)


# 标准化特征值
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM模型
svm_model = SVC(kernel='rbf')

# 训练模型
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy}")
print("分类报告:")
print(classification_report(y_test, y_pred, zero_division=0))

f1 = f1_score(y_test, y_pred,average="macro")

print(f1)

# 计算并绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=svm_model.classes_, columns=svm_model.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predict')
plt.ylabel('Actually')
plt.show()