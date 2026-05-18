import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Dataset
# =========================
df = pd.read_csv(
    "data/raw/credit_card_fraud_10k.csv",
    sep=';'
)

# =========================
# Informasi Dataset
# =========================
print("===== INFO DATASET =====")
print(df.info())

print("\n===== MISSING VALUE =====")
print(df.isnull().sum())

print("\n===== 5 DATA TERATAS =====")
print(df.head())

# =========================
# 1. Distribusi Fraud
# =========================
plt.figure(figsize=(6,4))

sns.countplot(x='is_fraud', data=df)

plt.title('Distribusi Fraud dan Non-Fraud')
plt.xlabel('Kelas Transaksi')
plt.ylabel('Jumlah')

plt.xticks([0,1], ['Non-Fraud', 'Fraud'])

plt.savefig('fraud_distribution.png')

plt.show()

# =========================
# 2. Histogram Amount
# =========================
plt.figure(figsize=(8,4))

sns.histplot(df['amount'], bins=50, kde=True)

plt.title('Histogram Transaction Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')

plt.savefig('amount_histogram.png')

plt.show()

# =========================
# 3. Heatmap Korelasi
# =========================
plt.figure(figsize=(12,8))

corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap='coolwarm', annot=True)

plt.title('Heatmap Korelasi Fitur')

plt.savefig('correlation_heatmap.png')

plt.show()

# =========================
# 4. Boxplot Amount
# =========================
plt.figure(figsize=(6,4))

sns.boxplot(x=df['amount'])

plt.title('Boxplot Transaction Amount')

plt.savefig('amount_boxplot.png')

plt.show()

# =========================
# 5. Distribusi Jam Transaksi
# =========================
plt.figure(figsize=(8,4))

sns.histplot(df['transaction_hour'], bins=24)

plt.title('Distribusi Jam Transaksi')

plt.xlabel('Jam')
plt.ylabel('Jumlah')

plt.savefig('transaction_hour_distribution.png')

plt.show()

print("\nEDA SELESAI!")
print("File gambar berhasil disimpan.")