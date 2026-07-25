import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from pathlib import Path
# pyrefly: ignore [missing-import]
import mlflow
import pandas as pd
# pyrefly: ignore [missing-import]
import mlflow.xgboost
# pyrefly: ignore [missing-import]
from mlflow.models import infer_signature
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str):
  """
  Trains an XGBoost model and logs with MLflow.

  Args:
    df (pd.DataFrame): Feature dataset.
    target_col (str): Name of the target column.

  Returns:
    model: Trained XGBClassifier
    X_test, y_test: Test split, untuk dipakai evaluasi lebih lanjut
  """
  # project_root: naik 2 folder dari src/models/train.py -> ke root project
  project_root = Path(__file__).resolve().parent.parent.parent
  mlruns_path = project_root / "mlruns"
  mlflow.set_tracking_uri(mlruns_path.as_uri())
  mlflow.set_experiment("Telco Churn")

  X = df.drop(columns=[target_col])
  y = df[target_col]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )

  model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
  )

  with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)

    # Log params, metrics
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("recall", rec)

    # Log model dengan signature eksplisit (hilangkan warning skema integer)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(model, name="model", signature=signature)

    # 🔑 Log dataset dengan source path yang valid (hilangkan warning ambiguitas source)
    train_ds = mlflow.data.from_pandas(
      df, source="data/processed/telco_churn_processed.csv"
    )
    mlflow.log_input(train_ds, context="training")

    print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")

  return model, X_test, y_test