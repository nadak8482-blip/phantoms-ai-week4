import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.95, 0.05],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_biased = LogisticRegression(random_state=42)
model_biased.fit(X_train, y_train)
y_pred_biased = model_biased.predict(X_test)

print('=' * 50)
print('1. الأداء قبل المعالجة (الخدعة):')
print('Accuracy:', accuracy_score(y_test, y_pred_biased))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_biased))
print('Classification Report:\n', classification_report(y_test, y_pred_biased))

model_balanced = LogisticRegression(
    class_weight='balanced', random_state=42
)
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)

print('=' * 50)
print('2. الأداء بعد المعالجة (Class Weights):')
print('Accuracy:', accuracy_score(y_test, y_pred_balanced))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_balanced))
print(
    'Classification Report:\n', classification_report(y_test, y_pred_balanced)
)
print('=' * 50)


https://colab.research.google.com/drive/1SeOigE1pABeQaskqKUfGTMt7B0uI3Zjs