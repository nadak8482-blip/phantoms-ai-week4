import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_lr = LogisticRegression(max_iter=5000, random_state=42)
scores_lr = cross_val_score(model_lr, X, y, cv=cv, scoring='accuracy')

model_dt = DecisionTreeClassifier(random_state=42)
scores_dt = cross_val_score(model_dt, X, y, cv=cv, scoring='accuracy')

print('=' * 50)
print('تقييم نموذج Logistic Regression عبر 5-Folds:')
print('Scores:', scores_lr)
print('Mean Accuracy:', np.mean(scores_lr))
print('Standard Deviation:', np.std(scores_lr))

print('=' * 50)
print('تقييم نموذج Decision Tree عبر 5-Folds:')
print('Scores:', scores_dt)
print('Mean Accuracy:', np.mean(scores_dt))
print('Standard Deviation:', np.std(scores_dt))
print('=' * 50)
