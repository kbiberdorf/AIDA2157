import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- PART 1: PROBLEM DEFINITION & DATA PREP ---
np.random.seed(101)
n_samples = 5000
X = np.random.randn(n_samples, 5)
y = np.zeros(n_samples)
fraud_indices = np.random.choice(n_samples, 100, replace=False)
y[fraud_indices] = 1

# SIGNAL FIX: Without this, all models will have 0% Recall because the data is identical.
# We shift the fraud features so the "Hunter" has a scent to follow.
X[fraud_indices] += 2.5 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# --- PART 2: THE ENSEMBLE COMPETITION ---
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=50, n_estimators=100, learning_rate=0.1, random_state=42)
}

results_data = []

# --- PART 3: COMPARISON & ANALYSIS ---
print(f"{'Model':<20} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results_data.append(f"{name:<20} | {acc:.4f}     | {rec:.4f}     | {f1:.4f}")
    print(results_data[-1])

# Task 3.2: Visualizing the "Holes"
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(models["XGBoost"], X_test, y_test, display_labels=["Normal", "FRAUD"], cmap='Reds', ax=ax)
plt.title("XGBoost Confusion Matrix: Identifying the 'Crashed Planes'")
plt.savefig('confusion_matrix.png')
plt.show()

# Calculate missed frauds for reasoning
xgb_preds = models["XGBoost"].predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, xgb_preds).ravel()

# --- PART 4: REASONING & FILE EXPORT ---
reasoning = f"""
PART 3.1: COMPARISON TABLE
{'-'*45}
Model                | Accuracy   | Recall     | F1-Score
{'-'*45}
{results_data[0]}
{results_data[1]}
{results_data[2]}

PART 4: REASONING QUESTIONS
1. The Imbalance Challenge: Why is 98% Accuracy a failure?
In this dataset, 98% of transactions are legitimate. A model that blindly predicts "Normal" for every single transaction would 
achieve 98% accuracy while catching 0% of fraud. In fraud detection, the goal is to stop the 2%, not to confirm the 98%.

2. Sequential Learning: How did XGBoost find the 2%?
XGBoost is a boosting algorithm that builds trees sequentially. Each new tree focuses specifically on the errors (residuals) 
made by the previous trees. By assigning a high 'scale_pos_weight', we forced the algorithm to treat missed frauds as massive errors, 
pushing the model to learn the specific patterns of the minority class.

3. The Refinement Trade-off (Bias vs. Variance):
A high learning rate (0.9) makes the model learn very quickly, which reduces Bias (training error drops). 
However, it makes the model highly sensitive to the specific noise in the training set, leading to High Variance (overfitting). 
This causes the test error to rise because the model fails to generalize to new data.

4. Survivorship Bias: How does Boosting prevent focusing only on "easy" cases?
Boosting focuses on the "difficult" cases—the ones the model is currently getting wrong. 
By re-weighting or focusing on misclassified points (the 'Crashed Planes'), it ensures the model doesn't just rest on its success 
with the easy, legitimate transactions (the 'Returning Planes').

TASK 3.2: THE HOLES
The XGBoost model missed {fn} frauds (False Negatives). These represent the 'Crashed Planes'—fraudulent transactions that were 
incorrectly labeled as legitimate and 'flew' past our defenses.
"""

with open('results.txt', 'w') as f:
    f.write(reasoning)

print("\n[SUCCESS] confusion_matrix.png and results.txt have been generated.")