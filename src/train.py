import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Membuat folder model
os.makedirs("models", exist_ok=True)

# Load data training hasil prepare.py
train_df = pd.read_csv("data/processed/train.csv")

target_column = "is_fraud"

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

print("Train data loaded:", X_train.shape, y_train.shape)

print("\nDistribusi class sebelum SMOTE:")
print(y_train.value_counts())

# SMOTE untuk menangani data imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nDistribusi class setelah SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Base model
base_models = [
    (
        "rf",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    ),
    (
        "knn",
        Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ])
    )
]

# Meta model
meta_model = LogisticRegression(max_iter=1000)

# Stacking Ensemble
model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    n_jobs=-1
)

# MLflow
mlflow.set_experiment("Credit Card Fraud Detection")

with mlflow.start_run(run_name="Stacking Ensemble with SMOTE"):
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_resampled)
    y_prob = model.predict_proba(X_resampled)[:, 1]

    accuracy = accuracy_score(y_resampled, y_pred)
    precision = precision_score(y_resampled, y_pred, zero_division=0)
    recall = recall_score(y_resampled, y_pred, zero_division=0)
    f1 = f1_score(y_resampled, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_resampled, y_prob)

    mlflow.log_param("model", "Stacking Ensemble")
    mlflow.log_param("sampling_method", "SMOTE")
    mlflow.log_param("target_column", target_column)
    mlflow.log_param("features_count", X_train.shape[1])

    mlflow.log_metric("train_accuracy", accuracy)
    mlflow.log_metric("train_precision", precision)
    mlflow.log_metric("train_recall", recall)
    mlflow.log_metric("train_f1_score", f1)
    mlflow.log_metric("train_roc_auc", roc_auc)

    joblib.dump(model, "models/stacking_model.pkl")
    mlflow.log_artifact("models/stacking_model.pkl")

print("\nTraining selesai!")
print("Model tersimpan di models/stacking_model.pkl")