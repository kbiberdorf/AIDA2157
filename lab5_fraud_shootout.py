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

# Initialize the competitors with specific Fraud-fighting settings
models = {
    "Single Tree (Weak)": DecisionTreeClassifier(max_depth=5),
    "Random Forest (Bagging)": RandomForestClassifier(n_estimators=100),
    "XGBoost (The Hunter)": XGBClassifier(scale_pos_weight=50, n_estimators=100)
}

# Run the shootout
f1_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate F1-Score instead of Accuracy
    f1 = f1_score(y_test, y_pred)
    f1_results[name] = f1
    
    print(f"--- {name} ---")
    print(f"F1-Score: {f1:.4f}")
    # This report shows us how many actual frauds were caught (Recall)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'FRAUD']))

# Visualize the Results (The F1-Score Jump)
plt.figure(figsize=(10, 6))
plt.bar(f1_results.keys(), f1_results.values(), color=['#ff9999', '#66b3ff', '#99ff99'])
plt.ylabel("F1-Score (Fraud Detection Power)")
plt.title("Fraud Shootout: Accuracy is a Lie, F1 is King")
plt.ylim(0, 1.0) 
plt.show()

from sklearn.metrics import recall_score, accuracy_score, ConfusionMatrixDisplay

# Dictionary to store our final stats
stats = {}

for name, model in models.items():
    preds = model.predict(X_test)
    stats[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds)
    }

# Print the comparison table
print(f"{'Model':<25} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 65)
for name, m in stats.items():
    print(f"{name:<25} | {m['Accuracy']:.4f}     | {m['Recall']:.4f}     | {m['F1-Score']:.4f}")