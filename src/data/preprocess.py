import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
  """
  Pembersihan dasar untuk data Telco Churn.
  - Merapikan nama kolom
  - Menghapus kolom ID yang jelas tidak diperlukan
  - Memperbaiki TotalCharges menjadi tipe numerik
  - Mengubah target Churn menjadi 0/1 jika diperlukan
  - Penanganan sederhana untuk nilai kosong (NA)
  """
  # Rapihkan header kolom
  df.columns = df.columns.str.strip() # Hapus spasi di awal/akhir nama kolom

  # Hapus kolom ID jika ada
  for col in ["customerID", "CustomerID", "customer_id"]:
    if col in df.columns:
      df = df.drop(columns=[col])

  # Ubah target menjadi 0/1 jika formatnya Yes/No
  if target_col in df.columns and df[target_col].dtype == "object":
    df[target_col] = df[target_col].str.strip().map({'No': 0, "Yes": 1})

  # TotalCharges sering memiliki nilaai kosong (blank) di datase ini -> paksa jadi float
  if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
  
  # SeniorCitizen harus berup integer 0/1 jika kolom nya ada
  if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

  # Strategi sederhana untuk menangani nilai kosong (NA):
  # - Kolom Numerik: isi dengan 0
  # - kolom lainnya: dibiarkan untuk ditangai oleh encoder (get_dummies aman terhadap NaN)
  num_cols = df.select_dtypes(include=["number"]).columns
  df[num_cols] = df[num_cols].fillna(0)

  return df