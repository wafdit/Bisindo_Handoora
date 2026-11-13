import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

DATASET_CSV = 'dataset_kata_level2.csv'
MODEL_PKL = 'model_kata_level2.pkl'

# 1. Load dataset dari CSV (3810 fitur urutan 30 frame)
df = pd.read_csv(DATASET_CSV, header=None)
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

print(f"Total sampel sekuens: {len(X)} | Jumlah kelas kata: {len(np.unique(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Latih Neural Network (MLP) pada rangkaian 30 frame
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=300,
    random_state=42
)
model.fit(X_train, y_train)

# 3. Evaluasi & Simpan
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Akurasi Validasi Level 2: {acc * 100:.2f}%")

with open(MODEL_PKL, 'wb') as f:
    pickle.dump(model, f)
print(f"Model tersimpan sebagai: {MODEL_PKL}")