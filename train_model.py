import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset CSV yang baru saja Anda rekam
csv_file = 'dataset_abjad_optimal.csv'
print(f"Membaca dataset dari {csv_file}...")
df = pd.read_csv(csv_file)

# 2. Pisahkan fitur (koordinat tangan) dan target (label huruf)
X = df.drop('label', axis=1)
y = df['label']

print(f"Total baris data: {len(df)}")
print(f"Daftar huruf yang direkam: {y.unique()}")

# 3. Bagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Latih model Random Forest
print("Sedang melatih model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.values, y_train)

# 5. Evaluasi akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")

# 6. Simpan model menjadi file .pkl
model_filename = 'model_abjad.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"Model berhasil disimpan sebagai: {model_filename}")