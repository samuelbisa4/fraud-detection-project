import os
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Folder output
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

target_column = "is_fraud"

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nDistribusi class sebelum SMOTE:")
print(y_train.value_counts())

# SMOTE hanya pada data training
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nDistribusi class setelah SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Model yang dibandingkan
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Stacking Ensemble": StackingClassifier(
        estimators=[
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
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1
    )
}

# MLflow experiment
mlflow.set_experiment("Credit Card Fraud Model Tracking")

best_model = None
best_model_name = None
best_f1 = -1

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print("\nTraining model:", model_name)

        # Training model
        model.fit(X_resampled, y_resampled)

        # Prediksi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log parameter
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("sampling_method", "SMOTE")
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("features_count", X_train.shape[1])

        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Simpan model setiap eksperimen
        model_filename = model_name.replace(" ", "_").lower()
        model_path = f"models/{model_filename}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Confusion matrix setiap model
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        cm_path = f"outputs/confusion_matrix_{model_filename}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(cm_path)

        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)
        print("ROC AUC  :", roc_auc)
        print("-" * 40)

        # Pilih model terbaik berdasarkan F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

# Simpan model terbaik
joblib.dump(best_model, "models/best_model.pkl")

with open("models/best_model_info.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best F1 Score: {best_f1}\n")

print("\nModel terbaik:", best_model_name)
print("Best F1 Score:", best_f1)
print("Model terbaik disimpan di models/best_model.pkl")