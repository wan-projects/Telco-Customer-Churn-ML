import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
  """
  Menerapkan binary encoding yang deterministik untuk fitur dengan 2 kategori.

  Fungsi ini mengimplementasikan logika inti binary encoding yang mengubah
  fitur kategorikal dengan tepat 2 nilai menjadi integer 0/1. Mapping-nya 
  bersifat deteministik dan harus konsisten antara training dan serving.
  """
  # Ambil nilai unik dan buang NaN
  vals = list(pd.Series(s.dropna().unique()).astype(str))
  valset = set(vals)

  # === MAPPING BINARY YANG DETERMINISTIK
  # PENTING: Mapping persis ini di-hardcode juga di serving pipeline

  # Mapping Yes/No (pola paling umum di data telekomunikasi)
  if valset == {'Yes', 'No'}:
    return s.map({"No": 0, "Yes": 1}).astype("Int64")

  # Mapping gender (fitur demografi)
  if valset == {"Male", "Female"}:
    return s.map({"Female": 0, "Male": 1}).astype("Int64")
  
  # === MAPPING BINARY GENERIK ===
  # Untuk fitur 2-kategorikal lainnya, gunakan urutan alfabetis yang stabil
  if len(vals) == 2:
    # Urutan nilai untuk memastikan mapping konsisten di setiap run
    sorted_vals = sorted(vals)
    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    return s.astype(str).map(mapping).astype("Int64")

  # === FITUR NON-BINARY ===
  # Kembalikan tanpa perubahan -  akan ditangani oleh one-hot encoding
  return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
  """
  Menerapkan pipeline feature engineering lengkap untuk data training.

  Ini adalah fungsi utama feature engineering yang mengubah data customer mentah
  menjadi fitur yang siap dipakai ML. Transformasi ini harus direplikasi persis
  di serving pipeline agar akurasi prediksi tetap terjaga. 
  """
  df = df.copy()
  print(f"🔧 Memulai feature engineering pada {df.shape[1]} kolom...")

  # === STEP 1: Identifikasi Tipe Fitur ===
  # Cari kolom kategorikal (dtype object), kecuali kolom target
  obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
  numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

  print (f"   📊 Ditemukan {len(obj_cols)} kolom kategorikal dan {len(numeric_cols)} kolom numerik")

  # === STEP 2: Pisahkan Ketegorikal Berdasarkan Kardinalitas ===
  # Fitur binary (tepat 2 nilai unik) mendapat binary encoding
  # Fitur multi-kategori (>2 nilai unik) mendapat one-hot encoding
  binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
  multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

  print(f"   🔢 Fitur binary: {len(binary_cols)} | Fitur multi-kategori: {len(multi_cols)}")
  if binary_cols:
    print(f"      Binary: {binary_cols}")
  if multi_cols:
    print(f"      Multi-kategori: {multi_cols}")

  # === STEP 3: Terapkan Binary Encoding ===
  # Ubah fitur 2-kategori menjadi 0/1 menggunakan mapping deterministik
  for c in binary_cols:
    original_dtype = df[c].dtype
    df[c] = _map_binary_series(df[c].astype(str))
    print(f"      ✅ {c}: {original_dtype} -> binary (0/1)")
  
  # === STEP 4: One-Hot Encoding untuk Fitur Multi-Kategori ===
  # PENTING: drop_first=True mencegah multikolinearitas
  if multi_cols:
    print(f"\n   🌟 Menerapkan one-hot encoding pada {len(multi_cols)} kolom multi-kategori...")
    original_shape = df.shape

    # Terapkan one-hot encoding dengan drop_first=True (sama seperti di serving)
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    new_features = df.shape[1] - original_shape[1] + len(multi_cols)
    print(f"      ✅ Membuat {new_features} fitur baru dari {len(multi_cols)} kolom kategorikal")

  # === STEP 5: Konversi Kolom Boolean ke Integer ===
  # XGBoost membutuhkan input integer, bukan boolean.
  # Dijalankan SETELAH one-hot encoding, supaya kolom bool hasil
  # pd.get_dummies() (mis. Contract_One year, InternetService_No, dll)
  # ikut tertangkap dan dikonversi, bukan hanya kolom bool yang sudah
  # ada sejak awal.
  bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
  if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"\n   🔄 Mengkonversi {len(bool_cols)} kolom boolean menjadi int: {bool_cols}")


  # === STEP 6: Pembersihan Tipe Data ===
  # Ubah nullable integer menjadi (Int64) menjadi integer standar untuk XGBoost
  for c in binary_cols:
    if pd.api.types.is_integer_dtype(df[c]):
      # Isi nilai NaN yang tersisa dengan 0 lalu konversi ke int
      df[c] = df[c].fillna(0).astype(int)
    
  print(f"\n✅ Feature engineering selesai: {df.shape[1]} fitur akhir")
  return df