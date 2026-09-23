import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

y_test = [0, 1, 2, 2, 0, 1, 1, 0, 2, 1]
y_pred = [0, 2, 2, 2, 0, 1, 0, 0, 2, 1]

print('=' * 50)
print('بدء عملية تقييم النموذج - اليوم الأول')
print('=' * 50)

cm = confusion_matrix(y_test, y_pred)
print('\n[1] Confusion Matrix Array:')
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Phantoms AI (W4D1)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print('\n[2] Classification Report (Precision, Recall, F1-Score):')
report = classification_report(y_test, y_pred)
print(report)

Untitled10.ipynb - Colab https://share.google/YWLZTTQazJVL1YadO