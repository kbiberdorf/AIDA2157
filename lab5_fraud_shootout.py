import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Generate Fraud Dataset (Anonymized Features V1-V5)
np.random.seed(101)
n_samples = 5000
X = np.random.randn(n_samples, 5)
# Target: 1 is Fraud, 0 is Normal. Fraud is only 2% of data.
y = np.zeros(n_samples)
fraud_indices = np.random.choice(n_samples, 100, replace=False)
y[fraud_indices] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)