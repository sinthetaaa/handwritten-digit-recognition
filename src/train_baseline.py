# train_baseline.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target 
X = X / 16.0

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(Xtr, ytr)

pred = clf.predict(Xte)
print("Accuracy:", accuracy_score(yte, pred))
print(classification_report(yte, pred))

ConfusionMatrixDisplay.from_predictions(yte, pred)
plt.title("Baseline Confusion Matrix")
plt.tight_layout()
plt.show()
