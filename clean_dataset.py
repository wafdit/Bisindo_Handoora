import pandas as pd

csv_file = 'dataset_abjad_2hands.csv'

# 1. Baca dataset
print(f"Membaca {csv_file}...")
df = pd.read_csv(csv_file)

print("Jumlah data sebelum dihapus:")
print(df['label'].value_counts())

# 2. Filter data: buang baris yang labelnya 'U'
df_cleaned = df[df['label'] != 'U']

# 3. Simpan kembali ke file CSV yang sama
df_cleaned.to_csv(csv_file, index=False)

print("\nData huruf 'U' berhasil dihapus!")
print("Jumlah data sekarang:")
print(df_cleaned['label'].value_counts())