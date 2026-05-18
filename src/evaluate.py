import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Folder output
os.makedirs("outputs", exist_ok=True)

# Load data test
test_df = pd.read_csv("data/processed/test.csv")

target_column = "is_fraud"

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Load model
model = joblib.load("models/stacking_model.pkl")

# Prediksi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Hitung metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)

# Simpan metrics
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Simpan classification report
with open("outputs/report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Stacking Ensemble")
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Stacking Ensemble")
plt.savefig("outputs/roc_curve.png", bbox_inches="tight")
plt.close()

# Bar chart metrics
plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1.05)
plt.title("Model Evaluation Metrics")
plt.ylabel("Score")

for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

plt.savefig("outputs/metrics_bar_chart.png", bbox_inches="tight")
plt.close()

print("\nEvaluasi selesai!")
print("Output disimpan di folder outputs/")