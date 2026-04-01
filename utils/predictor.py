import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_engine.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def preprocess_dataframe(df):
    original_df = df.copy()

    for col in ["class", "attack_class", "attack_type", "label", "difficulty_level", "difficulty", "target"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)

    df = df.apply(pd.to_numeric, errors="coerce")
    return original_df, df


def align_columns(df, expected_columns):
    df = df.copy()

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    extra_cols = [c for c in df.columns if c not in expected_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    return df[expected_columns]


def predict_dataframe(df):
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    expected_columns = joblib.load(FEATURE_COLUMNS_PATH)

    original_df, processed_df = preprocess_dataframe(df.copy())
    processed_df = align_columns(processed_df, expected_columns)

    X_scaled = pipeline.transform(processed_df)

    pred_encoded = model.predict(X_scaled)
    pred_probs = model.predict_proba(X_scaled).max(axis=1) * 100
    pred_labels = label_encoder.inverse_transform(pred_encoded)

    full_result_df = original_df.copy()
    full_result_df["predicted_class"] = pred_labels
    full_result_df["confidence"] = pred_probs.round(2)
    full_result_df["threat_status"] = full_result_df["predicted_class"].apply(
        lambda x: "Threat" if str(x).lower() != "normal" else "Normal"
    )

    useful_cols = []
    for col in [
        "duration",
        "src_bytes",
        "dst_bytes",
        "count",
        "srv_count",
        "predicted_class",
        "confidence",
        "threat_status"
    ]:
        if col in full_result_df.columns:
            useful_cols.append(col)

    display_result_df = full_result_df[useful_cols].copy()
    return full_result_df, display_result_df


def predict_from_file(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return predict_dataframe(df)