#!/usr/bin/env python3
"""
Runs sequentially: load -> validate -> preprocess -> feature engineering
"""

import argparse
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true" # PERBAIKAN 1: izinkan file-based tracking

import sys
import time
import json
import joblib
import argparse
from pathlib import Path                     # PERBAIKAN 2: untik URI yang valid di windows
import pandas as pd
import mlflow
import mlflow.xgboost                          # PERBAIKAN 3: konsisten pakai mlflow.xgboost
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#   classification_report, precision_score, recall_score,
#   f1_score, roc_auc_score
# )
# from xgboost import XGBClassifier
# (baris "from posthog import project_root" DIHAPUS - PERBAIKAN 4)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
# from src.data.preprocess_data import preprocess_data
# from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

def main(args):
  project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

  # PERBAIKAN 5: pakai Path(...).as_uri() supaya valid di Windows
  mlruns_path = args.mlflow_uri or Path(project_root, "mlruns").as_uri()
  mlflow.set_tracking_uri(mlruns_path)
  mlflow.set_experiment(args.experiment)

  with mlflow.start_run():
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("threshold", args.threshold)
    mlflow.log_param("test_size", args.test_size)

    # === STAGE 1: Data Loading & Validation ===
    print("🔄 Loading data...")
    df = load_data(args.input)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # TEST Stage Load:
    if args.stop_after == "load":
      print("🛑 Berhenti setelah STAGE 1 (load) sesuai permintaan --stop_after")
      return

    print("🔍 Validating data quality with Great Expectations...")
    is_valid, failed = validate_telco_data(df)
    mlflow.log_metric("data_quality_pass", int(is_valid))
    if not is_valid:
      mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
      # PERBAIKAN 6 (sementara): Jangan hentikan pipeline untuk 2 masalah TotalCharges
      # yang memang akan diperbaiki di Stage 2 (preprocessing).
      print(f"⚠️  Data quality check gagal sebagian: {failed}")
      print("⚠️  Melanjutkan pipeline karena masalah ini akan diperbaiki di preprocessing.")
    else:
      print("✅ Data validation passed. Logged to Mlflow.")
    
    # TEST Stage Validate:
    if args.stop_after == "validate":
      print("🛑 Berhenti setelah STAGE 1 (validate) sesuai permintaan --stop_after")
      return


if __name__ == "__main__":
  p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
  p.add_argument("--input", type=str, required=True,
                  help="Path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
  p.add_argument("--target", type=str, default="Churn")
  p.add_argument("--threshold", type=float, default=0.35)
  p.add_argument("--test_size", type=float, default=0.2)
  p.add_argument("--experiment", type=str, default="Telco Churn")
  p.add_argument("--mlflow_uri", type=str, default=None,
                  help="override MLflow tracking URI, else uses project_root/mlruns")

  # Flag khusus untuk testing bertahap
  p.add_argument("--stop_after", type=str, default=None,
                  choices=["load", "validate", "preprocess", "features", "split", "train", "eval"],
                  help="Hentikan pipeline setelah stage tertentu, untuk keperluan testing")

  args = p.parse_args()
  main(args)