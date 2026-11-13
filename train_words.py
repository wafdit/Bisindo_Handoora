import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = 'dataset_kata'
actions = np.array(sorted(os.listdir(DATA_PATH)))
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
print("Daftar kelas kata:", label_map)

sequences, labels = [], []
for action in actions:
    action_dir = os.path.join(DATA_PATH, action)
    for file_name in os.listdir(action_dir):
        if file_name.endswith('.npy'):
            res = np.load(os.path.join(action_dir, file_name))
            sequences.append(res)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=labels
)

# Arsitektur LSTM ringan & cepat untuk hackathon
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 127)),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=15, restore_best_weights=True)

print("Melatih model LSTM...")
model.fit(
    X_train, y_train,
    epochs=120,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nAkurasi Model Kata: {acc * 100:.2f}%")

# Simpan model & daftar label
model.save('model_kata.h5')
np.save('label_kata.npy', actions)
print("Model tersimpan sebagai 'model_kata.h5' dan 'label_kata.npy'")