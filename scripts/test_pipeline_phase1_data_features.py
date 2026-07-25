# test_pipeline_phase1.py
import os
import sys

# Pastikan Python bisa menemukan package src nya
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# === Konfigurasi
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

def main():
  print("=== Menguji Phase 1: Load -> Preprocess -> Build Features ===")
  
  # 1. Load Data
  print("\n[1] Loading data...")
  df = load_data(DATA_PATH)
  print(f"Data loaded. Shape: {df.shape}")
  print(df.head())

  # 2. Preprocess
  print("\n[2] Preprocessing data...")
  df_clean = preprocess_data(df, target_col=TARGET_COL)
  print(f"Data setelah preprocessing. Shape: {df_clean.shape}")
  print(df_clean.head())

  # 3. Build Feature
  print("\n[3] Melakukan feature engineering...")
  df_processed = build_features(df_clean, target_col=TARGET_COL)
  print(f"Data setelah feature engineering. Shape: {df_processed.shape}")
  print(df_processed.head())

  print("\n✅ Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()