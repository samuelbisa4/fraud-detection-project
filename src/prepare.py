import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Membuat folder output
os.makedirs("data/processed", exist_ok=True)

# Load dataset baru
# Dataset ini menggunakan delimiter titik koma (;)
df = pd.read_csv("data/raw/credit_card_fraud_10k.csv", sep=";")

print("Data shape:", df.shape)
print("Kolom dataset:")
print(df.columns.tolist())

# Target / label
target_column = "is_fraud"

# Cek apakah target tersedia
if target_column not in df.columns:
    raise ValueError("Kolom target 'is_fraud' tidak ditemukan.")

# Pisahkan fitur dan target
# transaction_id dibuang karena hanya ID, bukan fitur prediksi
X = df.drop(columns=[target_column, "transaction_id"], errors="ignore")
y = df[target_column]

# Encoding fitur kategorikal
# merchant_category berupa teks, jadi harus diubah ke numerik
X = pd.get_dummies(X, columns=["merchant_category"], drop_first=False)

print("\nJumlah fitur setelah encoding:", X.shape[1])
print("Nama fitur:")
print(X.columns.tolist())

print("\nDistribusi class sebelum split:")
print(y.value_counts())

# Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Gabungkan kembali fitur dan target
train_df = X_train.copy()
train_df[target_column] = y_train

test_df = X_test.copy()
test_df[target_column] = y_test

# Simpan hasil preprocessing
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nPrepare selesai!")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nDistribusi class train:")
print(y_train.value_counts())

print("\nDistribusi class test:")
print(y_test.value_counts())