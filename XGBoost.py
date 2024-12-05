import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#数据集的获取
diabetes=pd.read_csv("./data.csv")
features=['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']

#数据集的划分
x_train,x_test,y_train,y_test=train_test_split(diabetes[features],diabetes['target'],random_state=100,test_size=0.3)

#特征工程
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.fit_transform(x_test)

#设置xgboost的参数
params={'learning_rate':0.3,
        'max_depth':3,
        'objective':'multi:softmax',
        'eval_metric':'mlogloss',
        'num_class':3
}

#将测试集、训练集转化为DMatrix格式
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test,label=y_test)

#机器学习
model=xgb.train(params,dtrain,num_boost_round=100,evals=[(dtest,'mlogloss')],early_stopping_rounds=10)

#模型评估
y_pred=model.predict(dtest)
print(y_pred)
f1_score=f1_score(y_test,y_pred,average='macro')
print('f1_score =',f1_score)
accuracy_score=accuracy_score(y_test,y_pred)
print('accuracy_score =',accuracy_score)