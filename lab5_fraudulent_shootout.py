import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, ConfusionMatrixDisplay
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

# Train Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=50, random_state=42)
}

# Compare models
print(f"{'Model':<20} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"{name:<20} | {rec:.4f}     | {f1:.4f}")

# Using the cleaner built-in display
ConfusionMatrixDisplay.from_estimator(models["XGBoost (The Hunter)"], X_test, y_test, cmap='Reds')
plt.title("XGBoost Confusion Matrix")
plt.show()
