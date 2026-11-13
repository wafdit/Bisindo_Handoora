import pandas as pd

DATASET_CSV = 'dataset_kata_level2.csv'
HAPUS_X_TERAKHIR = 1  # Ubah angka ini: mau menghapus berapa baris terakhir?

df = pd.read_csv(DATASET_CSV, header=None)
jumlah_awal = len(df)

# Ambil data dari awal hingga dipotong X baris dari belakang
df_bersih = df.iloc[:-HAPUS_X_TERAKHIR]

df_bersih.to_csv(DATASET_CSV, index=False, header=False)

print(f"BERHASIL menghapus {HAPUS_X_TERAKHIR} sampel terakhir!")
print(f"Sisa data di file: {len(df_bersih)} baris (Awalnya {jumlah_awal}).")