import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score,accuracy_score
from sklearn.model_selection import cross_val_score, KFold

data = pd.read_csv('./data.csv')
x = data.drop('target', axis=1)
y = data['target']

frac_to_use = 0.06
indices = np.random.choice(x.index, size=int(frac_to_use * len(x)), replace=False)
x_small = x.loc[indices]
y_small = y.loc[indices]

x_train, x_test, y_train, y_test = train_test_split(x_small, y_small, test_size=0.2, random_state=142, stratify=y_small)

def optuna_objective(trial):
    max_depth = trial.suggest_int('max_depth', 5, 30)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    n_estimators = trial.suggest_int('n_estimators', 10, 50)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    gbc = GradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        random_state=142
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=142)
    cv_scores = cross_val_score(gbc, x_train, y_train, cv=kf, scoring=make_scorer(f1_score, average='macro'))
    mean_cv_score = np.mean(cv_scores)
    
    return mean_cv_score
def optimizer_optuna(n_trials):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)
    print(study.best_trial.params, study.best_trial.value)
    return study.best_trial.params, study.best_trial.value
best_params, best_score = optimizer_optuna(25)
best_model = GradientBoostingClassifier(**best_params, random_state=142)
best_model.fit(x_train, y_train)

y_pred = best_model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:")
print(accuracy)
print("\nF1 Score (macro):")
print(f1)
print("\nClassification Report:")
print(class_report)
