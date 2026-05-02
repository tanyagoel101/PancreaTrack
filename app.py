from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None


DATA_PATH = Path("data/archive/Debernardi et al 2020 data.csv")
DISCLAIMER = (
    "This tool is for research and educational purposes only and does not provide medical diagnosis."
)
MODEL_FEATURES = [
    "age",
    "sex",
    "creatinine",
    "LYVE1",
    "REG1B",
    "TFF1",
    "plasma_CA19_9",
]
NUMERIC_FEATURES = ["age", "creatinine", "LYVE1", "REG1B", "TFF1", "plasma_CA19_9"]
CATEGORICAL_FEATURES = ["sex"]
DISPLAY_NAMES = {
    "age": "Age",
    "sex_F": "Sex: Female",
    "sex_M": "Sex: Male",
    "creatinine": "Creatinine",
    "LYVE1": "LYVE1",
    "REG1B": "REG1B",
    "TFF1": "TFF1",
    "plasma_CA19_9": "CA19-9",
    "weight_loss_pct": "Weight loss",
    "glucose_or_hba1c": "Glucose/HbA1c",
    "new_onset_diabetes": "New-onset diabetes",
    "pain": "Abdominal or back pain",
    "jaundice": "Jaundice",
    "fatigue": "Fatigue",
}


@dataclass
class PreparedData:
    dataframe: pd.DataFrame
    target: pd.Series
    counts: dict[str, int]


def inject_theme() -> None:
    st.set_page_config(page_title="PancreaTrack", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        :root {
            --bg-main: #07131f;
            --bg-panel: rgba(11, 29, 47, 0.88);
            --bg-panel-strong: rgba(9, 23, 39, 0.96);
            --text-main: #f3f8ff;
            --text-soft: #bed1e6;
            --line: rgba(125, 199, 255, 0.16);
            --teal: #65d8c6;
            --cyan: #7fd7ff;
            --amber: #ffd67a;
            --rose: #ff8e8e;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(63, 214, 197, 0.14), transparent 24%),
                radial-gradient(circle at left top, rgba(80, 166, 255, 0.14), transparent 30%),
                linear-gradient(180deg, #04101b 0%, #091726 48%, #05111c 100%);
            color: var(--text-main);
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3, h4 {
            font-family: 'Manrope', sans-serif !important;
            letter-spacing: -0.02em;
        }
        p, li, label, .stCaption {
            color: var(--text-soft);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5, 17, 29, 0.98), rgba(8, 24, 40, 0.98));
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(10, 28, 45, 0.94), rgba(8, 21, 36, 0.92));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.2);
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-soft);
        }
        [data-testid="stMetricValue"] {
            font-family: 'Manrope', sans-serif;
            color: var(--text-main);
        }
        .metric-card, .info-card, .alert-card, .recommend-card, .hero-card, .section-card {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            background: var(--bg-panel);
            box-shadow: 0 18px 30px rgba(0, 0, 0, 0.16);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            padding: 1.35rem 1.4rem;
            background:
                linear-gradient(135deg, rgba(20, 54, 85, 0.95), rgba(8, 21, 36, 0.96)),
                radial-gradient(circle at top right, rgba(101, 216, 198, 0.18), transparent 30%);
        }
        .metric-card h4, .info-card h4, .alert-card h4, .recommend-card h4, .hero-card h4, .section-card h4 {
            margin: 0 0 0.5rem 0;
            color: var(--cyan);
        }
        .risk-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .risk-low { background: rgba(57, 181, 129, 0.18); color: #8cf0c2; }
        .risk-moderate { background: rgba(255, 194, 77, 0.18); color: #ffe09e; }
        .risk-high { background: rgba(255, 107, 107, 0.2); color: #ffabab; }
        .disclaimer {
            border-left: 4px solid var(--cyan);
            padding: 0.95rem 1rem;
            background: rgba(12, 35, 58, 0.82);
            border-radius: 14px;
        }
        .small-note {
            color: var(--text-soft);
            font-size: 0.94rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            color: #8fb8d7;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.7fr 1fr;
            gap: 1rem;
            align-items: stretch;
        }
        .stat-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.85rem;
        }
        .mini-stat {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
        }
        .mini-stat strong {
            display: block;
            color: var(--text-main);
            font-size: 1.15rem;
            margin-top: 0.15rem;
        }
        [data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: rgba(8, 24, 40, 0.64);
            border: 1px solid var(--line);
            padding: 0.35rem;
            border-radius: 14px;
        }
        button[role="tab"] {
            border-radius: 10px !important;
            padding: 0.55rem 0.9rem !important;
            color: var(--text-soft) !important;
            font-weight: 600 !important;
        }
        button[role="tab"][aria-selected="true"] {
            background: rgba(101, 216, 198, 0.15) !important;
            color: var(--text-main) !important;
        }
        .stAlert {
            border-radius: 14px;
        }
        .stDataFrame, div[data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_style(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,18,30,0.35)",
        font=dict(color="#e9f2ff", family="IBM Plex Sans, sans-serif"),
        title_font=dict(family="Manrope, sans-serif", size=20, color="#f3f8ff"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            bgcolor="rgba(6,18,30,0.0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#bed1e6")
    fig.update_yaxes(gridcolor="rgba(126, 171, 209, 0.12)", zeroline=False, color="#bed1e6")
    if height is not None:
        fig.update_layout(height=height)
    return fig


@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(data), encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str, encoding="utf-8-sig")


def load_data() -> pd.DataFrame | None:
    if DATA_PATH.exists():
        return _read_csv_from_path(str(DATA_PATH))

    st.warning("Dataset not found at the default path. Upload the Kaggle CSV to continue.")
    uploaded_file = st.file_uploader("Upload the urinary biomarker dataset CSV", type=["csv"])
    if uploaded_file is not None:
        return _read_csv_from_bytes(uploaded_file.getvalue())
    return None


def clean_data(raw_df: pd.DataFrame) -> PreparedData:
    df = raw_df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    numeric_columns = [
        "age",
        "plasma_CA19_9",
        "creatinine",
        "LYVE1",
        "REG1B",
        "TFF1",
        "REG1A",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "": np.nan})
    )
    df["sex"] = df["sex"].where(df["sex"].isin(["M", "F"]), np.nan)

    diagnosis_map = {1: 0, 2: 1, 3: 2}
    df["diagnosis_original"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df["diagnosis"] = df["diagnosis_original"].map(diagnosis_map)
    df = df.dropna(subset=["diagnosis", "age"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["is_pdac"] = (df["diagnosis"] == 2).astype(int)

    counts = {
        "control": int((df["diagnosis"] == 0).sum()),
        "benign": int((df["diagnosis"] == 1).sum()),
        "pdac": int((df["diagnosis"] == 2).sum()),
        "total": int(len(df)),
    }

    return PreparedData(dataframe=df, target=df["is_pdac"], counts=counts)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    try:
        categorical_encoder.set_output(transform="default")
    except Exception:
        pass

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", categorical_encoder),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "predicted_probabilities": y_prob,
    }


@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame) -> dict[str, Any]:
    feature_frame = df[MODEL_FEATURES].copy()
    target = df["is_pdac"]

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    logistic_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    random_forest_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=7,
                    min_samples_leaf=3,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }
    metrics: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics[model_name] = compute_metrics(y_test, predictions, probabilities)
        trained_models[model_name] = pipeline

    best_model_name = max(metrics, key=lambda item: metrics[item]["roc_auc"])
    return {
        "models": trained_models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    cleaned = []
    for name in feature_names:
        label = name.replace("num__", "").replace("cat__", "")
        cleaned.append(label)
    return cleaned


def calculate_shap_values(
    model_name: str,
    pipeline: Pipeline,
    x_reference: pd.DataFrame,
    patient_frame: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    if shap is None:
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_reference = preprocessor.transform(x_reference)
    feature_names = get_feature_names(pipeline)

    try:
        if model_name == "Random Forest":
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_reference)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.LinearExplainer(classifier, transformed_reference)
            shap_values = explainer.shap_values(transformed_reference)
        result: dict[str, Any] = {
            "feature_names": feature_names,
            "summary_values": np.abs(np.asarray(shap_values)).mean(axis=0),
        }
        if patient_frame is not None:
            patient_transformed = preprocessor.transform(patient_frame)
            patient_values = explainer.shap_values(patient_transformed)
            if isinstance(patient_values, list):
                patient_values = patient_values[1]
            result["patient_values"] = np.asarray(patient_values)[0]
        return result
    except Exception:
        return None


def fallback_patient_contributions(
    model_name: str,
    pipeline: Pipeline,
    patient_frame: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    transformed_patient = pipeline.named_steps["preprocessor"].transform(patient_frame)
    transformed_array = np.asarray(transformed_patient)[0]
    feature_names = get_feature_names(pipeline)
    classifier = pipeline.named_steps["classifier"]

    if model_name == "Logistic Regression":
        weights = classifier.coef_[0]
        contribution_values = transformed_array * weights
    else:
        importances = classifier.feature_importances_
        transformed_reference = pipeline.named_steps["preprocessor"].transform(reference_df)
        reference_center = np.asarray(transformed_reference).mean(axis=0)
        contribution_values = (transformed_array - reference_center) * importances

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contribution_values,
            "abs_contribution": np.abs(contribution_values),
        }
    ).sort_values("abs_contribution", ascending=False)
    frame["display_feature"] = frame["feature"].map(lambda val: DISPLAY_NAMES.get(val, val))
    return frame


def overlay_clinical_signals(patient_inputs: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    signals: list[tuple[str, float]] = []
    alerts: list[str] = []

    glucose_value = float(patient_inputs["glucose_or_hba1c"])
    if glucose_value >= 126:
        signals.append(("glucose_or_hba1c", 0.18))
        alerts.append("⚠️ Rising glucose or diabetes-range value may indicate metabolic change.")
    elif glucose_value >= 100:
        signals.append(("glucose_or_hba1c", 0.08))

    if patient_inputs["new_onset_diabetes"]:
        signals.append(("new_onset_diabetes", 0.14))
        alerts.append("⚠️ New-onset diabetes increases concern when paired with pancreatic biomarkers.")

    if float(patient_inputs["weight_loss_pct"]) > 5:
        signals.append(("weight_loss_pct", 0.16))
        alerts.append("⚠️ Weight loss greater than 5% is a notable warning trend.")

    if patient_inputs["jaundice"]:
        signals.append(("jaundice", 0.18))
        alerts.append("⚠️ Jaundice is a high-priority symptom signal that warrants prompt attention.")

    if patient_inputs["pain"]:
        signals.append(("pain", 0.09))

    if patient_inputs["fatigue"]:
        signals.append(("fatigue", 0.05))

    if not signals:
        signals.append(("glucose_or_hba1c", 0.0))

    signal_frame = pd.DataFrame(signals, columns=["feature", "weight"])
    signal_frame["display_feature"] = signal_frame["feature"].map(
        lambda val: DISPLAY_NAMES.get(val, val)
    )
    signal_frame = signal_frame.sort_values("weight", ascending=False)
    return alerts, signal_frame


def build_patient_frame(patient_inputs: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": patient_inputs["age"],
                "sex": patient_inputs["sex"],
                "creatinine": patient_inputs["creatinine"],
                "LYVE1": patient_inputs["LYVE1"],
                "REG1B": patient_inputs["REG1B"],
                "TFF1": patient_inputs["TFF1"],
                "plasma_CA19_9": patient_inputs["CA19_9"],
            }
        ]
    )


def risk_category(score: float) -> str:
    if score < 0.30:
        return "Low"
    if score < 0.70:
        return "Moderate"
    return "High"


def patient_recommendation(category: str) -> str:
    if category == "Low":
        return "Continue monitoring and review trends over time."
    if category == "Moderate":
        return "Follow up clinically and consider repeating labs to evaluate progression."
    return "Consider imaging and specialist referral based on the combined warning pattern."


def patient_narrative(
    contribution_frame: pd.DataFrame,
    overlay_frame: pd.DataFrame,
    score: float,
) -> str:
    model_drivers = contribution_frame.head(3)["display_feature"].tolist()
    overlay_drivers = overlay_frame[overlay_frame["weight"] > 0].head(2)["display_feature"].tolist()
    driver_text = ", ".join(model_drivers) if model_drivers else "the urinary biomarker profile"

    if overlay_drivers:
        overlay_text = ", ".join(overlay_drivers)
        return (
            f"This patient's elevated risk score of {score:.2f} is primarily influenced by "
            f"{driver_text}, with additional concern from {overlay_text}."
        )
    return (
        f"This patient's risk score of {score:.2f} is primarily influenced by {driver_text}."
    )


def predict_patient_risk(
    patient_inputs: dict[str, Any],
    trained_bundle: dict[str, Any],
) -> dict[str, Any]:
    model_name = trained_bundle["best_model_name"]
    pipeline = trained_bundle["best_model"]
    patient_frame = build_patient_frame(patient_inputs)
    base_probability = float(pipeline.predict_proba(patient_frame)[0, 1])

    overlay_alerts, overlay_frame = overlay_clinical_signals(patient_inputs)
    overlay_adjustment = float(overlay_frame["weight"].sum())
    blended_score = float(np.clip(base_probability + overlay_adjustment * 0.25, 0, 1))
    category = risk_category(blended_score)

    shap_result = calculate_shap_values(
        model_name=model_name,
        pipeline=pipeline,
        x_reference=trained_bundle["x_train"],
        patient_frame=patient_frame,
    )

    if shap_result and "patient_values" in shap_result:
        contribution_frame = pd.DataFrame(
            {
                "feature": shap_result["feature_names"],
                "contribution": shap_result["patient_values"],
            }
        )
        contribution_frame["abs_contribution"] = contribution_frame["contribution"].abs()
        contribution_frame["display_feature"] = contribution_frame["feature"].map(
            lambda val: DISPLAY_NAMES.get(val, val)
        )
        contribution_frame = contribution_frame.sort_values("abs_contribution", ascending=False)
    else:
        contribution_frame = fallback_patient_contributions(
            model_name=model_name,
            pipeline=pipeline,
            patient_frame=patient_frame,
            reference_df=trained_bundle["x_train"],
        )

    narrative = patient_narrative(contribution_frame, overlay_frame, blended_score)

    return {
        "base_probability": base_probability,
        "risk_score": blended_score,
        "risk_category": category,
        "recommendation": patient_recommendation(category),
        "contributions": contribution_frame,
        "overlay_alerts": overlay_alerts,
        "overlay_frame": overlay_frame,
        "narrative": narrative,
    }


def generate_simulated_trends(
    patient_inputs: dict[str, Any],
    risk_score: float,
) -> tuple[pd.DataFrame, list[str]]:
    seed = int(
        patient_inputs["age"] * 11
        + patient_inputs["creatinine"] * 100
        + patient_inputs["LYVE1"] * 10
        + patient_inputs["REG1B"]
    ) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=6, freq="ME")
    trend_length = len(months)
    risk_factor = 0.35 + risk_score

    current_weight = max(45.0, 78.0 * (1 - float(patient_inputs["weight_loss_pct"]) / 100.0))
    baseline_weight = current_weight / max(0.55, 1 - float(patient_inputs["weight_loss_pct"]) / 100.0)

    def build_series(current_value: float, upward_bias: float, noise_scale: float) -> np.ndarray:
        baseline = current_value / max(0.45, 1 + upward_bias)
        ramp = np.linspace(baseline, current_value, trend_length)
        noise = rng.normal(0, noise_scale, trend_length)
        return np.maximum(ramp + noise, 0)

    ca19_current = float(patient_inputs["CA19_9"])
    glucose_current = float(patient_inputs["glucose_or_hba1c"])
    frame = pd.DataFrame(index=range(trend_length))
    frame["month"] = months.to_list()
    frame["CA19-9"] = build_series(ca19_current, 0.25 * risk_factor, max(1.2, ca19_current * 0.02))
    frame["Glucose/HbA1c"] = build_series(
        glucose_current,
        0.12 * risk_factor + (0.1 if patient_inputs["new_onset_diabetes"] else 0),
        max(0.6, glucose_current * 0.015),
    )
    frame["Weight"] = np.linspace(baseline_weight, current_weight, trend_length) + rng.normal(
        0,
        0.35,
        trend_length,
    )
    frame["LYVE1"] = build_series(
        float(patient_inputs["LYVE1"]),
        0.18 * risk_factor,
        max(0.05, float(patient_inputs["LYVE1"]) * 0.02),
    )
    frame["REG1B"] = build_series(
        float(patient_inputs["REG1B"]),
        0.2 * risk_factor,
        max(0.4, float(patient_inputs["REG1B"]) * 0.018),
    )
    frame["TFF1"] = build_series(
        float(patient_inputs["TFF1"]),
        0.22 * risk_factor,
        max(1.0, float(patient_inputs["TFF1"]) * 0.02),
    )

    alerts: list[str] = []
    recent_ca = frame["CA19-9"].tail(3).to_numpy()
    if recent_ca[0] > 0 and (recent_ca[-1] - recent_ca[0]) / recent_ca[0] > 0.25:
        alerts.append("⚠️ Rapid increase in CA19-9 over the last 3 months.")

    weight_drop = (frame["Weight"].iloc[0] - frame["Weight"].iloc[-1]) / frame["Weight"].iloc[0]
    if weight_drop > 0.05:
        alerts.append("⚠️ Weight loss exceeds 5% across the simulated timeline.")

    glucose_series = frame["Glucose/HbA1c"].to_numpy()
    if glucose_series[-1] - glucose_series[0] > max(8, glucose_series[0] * 0.08):
        alerts.append("⚠️ Rising glucose trend may reflect new-onset diabetes physiology.")

    rising_markers = 0
    for marker in ["CA19-9", "LYVE1", "REG1B", "TFF1"]:
        values = frame[marker].to_numpy()
        if values[0] > 0 and (values[-1] - values[0]) / values[0] > 0.12:
            rising_markers += 1
    if rising_markers >= 3:
        alerts.append("⚠️ Multiple biomarker increases strengthen the need for follow-up.")

    return frame, alerts


def render_overview(prepared: PreparedData, trained_bundle: dict[str, Any]) -> None:
    best_model_name = trained_bundle["best_model_name"]
    best_metrics = trained_bundle["metrics"][best_model_name]

    st.subheader("PancreaTrack Overview")
    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="hero-card">
                <div class="eyebrow">Educational Clinical Decision Support</div>
                <h4>Interpretable early-risk signal dashboard</h4>
                <p>
                    PancreaTrack blends urinary biomarkers, symptom context, diabetes signals,
                    and simulated trend monitoring to surface weak signals that may be easy to miss
                    in a cross-sectional review.
                </p>
                <div class="stat-strip">
                    <div class="mini-stat">Selected model<strong>{best_model_name}</strong></div>
                    <div class="mini-stat">ROC-AUC<strong>{best_metrics['roc_auc']:.3f}</strong></div>
                    <div class="mini-stat">Threshold<strong>0.50</strong></div>
                </div>
            </div>
            <div class="section-card">
                <div class="eyebrow">What this app emphasizes</div>
                <h4>Signal aggregation over single-test diagnosis</h4>
                <p>Pancreatic cancer rarely presents with one decisive early marker. The dashboard focuses on trajectory, clustering, and explainability.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="disclaimer"><strong>Disclaimer:</strong> {DISCLAIMER}</div>', unsafe_allow_html=True)

    row_one = st.columns(4)
    row_one[0].metric("Total patients", prepared.counts["total"])
    row_one[1].metric("PDAC cases", prepared.counts["pdac"])
    row_one[2].metric("Benign cases", prepared.counts["benign"])
    row_one[3].metric("Controls", prepared.counts["control"])

    row_two = st.columns(3)
    row_two[0].metric("Best model", best_model_name)
    row_two[1].metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
    row_two[2].metric(
        "Sensitivity / Specificity",
        f"{best_metrics['sensitivity']:.2f} / {best_metrics['specificity']:.2f}",
    )

    diagnosis_counts = (
        prepared.dataframe["diagnosis"]
        .map({0: "Control", 1: "Benign", 2: "PDAC"})
        .value_counts()
        .rename_axis("group")
        .reset_index(name="count")
    )
    fig = px.bar(
        diagnosis_counts,
        x="count",
        y="group",
        orientation="h",
        color="group",
        color_discrete_map={"Control": "#59d0ff", "Benign": "#ffd166", "PDAC": "#ff6b6b"},
        title="Dataset composition",
    )
    fig.update_traces(texttemplate="%{x}", textposition="outside")
    apply_plot_style(fig, height=360)
    fig.update_layout(coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_patient_dashboard(
    patient_inputs: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    st.subheader("Patient Dashboard")
    st.caption("Educational clinical decision-support only. This is not a diagnostic result.")

    risk_class = prediction["risk_category"].lower()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="eyebrow">Patient-level output</div>
            <h4>Predicted PDAC risk score</h4>
            <p style="font-size:2.4rem;margin:0.25rem 0 0.5rem 0;font-family:'Manrope',sans-serif;"><strong>{prediction['risk_score']:.2f}</strong></p>
            <span class="risk-pill risk-{risk_class}">{prediction['risk_category']} risk</span>
            <p class="small-note" style="margin-top:0.85rem;">
                Model-only PDAC probability: {prediction['base_probability']:.2f}. Extra clinical signals are shown as a transparent overlay.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Risk category", prediction["risk_category"])
    kpi_cols[1].metric("Model-only PDAC probability", f"{prediction['base_probability']:.2f}")
    kpi_cols[2].metric("Clinical overlay signals", len(prediction["overlay_alerts"]))

    summary_cols = st.columns([1.45, 1])
    with summary_cols[0]:
        st.markdown(
            f"""
            <div class="recommend-card">
                <div class="eyebrow">Interpretation</div>
                <h4>Clinical interpretation</h4>
                <p>{prediction['narrative']}</p>
                <p><strong>Recommendation:</strong> {prediction['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with summary_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <div class="eyebrow">Current profile</div>
                <h4>Entered profile</h4>
                <p>Age: {age}</p>
                <p>Sex: {sex}</p>
                <p>CA19-9: {ca19}</p>
                <p>Glucose/HbA1c: {glucose}</p>
                <p>Weight loss: {weight_loss}%</p>
            </div>
            """.format(
                age=patient_inputs["age"],
                sex=patient_inputs["sex"],
                ca19=f"{patient_inputs['CA19_9']:.1f}",
                glucose=f"{patient_inputs['glucose_or_hba1c']:.1f}",
                weight_loss=f"{patient_inputs['weight_loss_pct']:.1f}",
            ),
            unsafe_allow_html=True,
        )

    if prediction["overlay_alerts"]:
        st.markdown("#### Active warning signals")
        for alert in prediction["overlay_alerts"]:
            st.error(alert)

    contribution_plot = prediction["contributions"].head(6).sort_values("contribution")
    contrib_fig = px.bar(
        contribution_plot,
        x="contribution",
        y="display_feature",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#65d8c6", "#ffd67a", "#ff8e8e"],
        title="Top feature contributions",
    )
    apply_plot_style(contrib_fig, height=380)
    contrib_fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(contrib_fig, use_container_width=True)


def render_biomarker_trends(
    patient_inputs: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    st.subheader("Biomarker Trends")
    st.caption("The source dataset is cross-sectional. These 6-month trajectories are simulated to illustrate pattern monitoring.")

    trend_df, alerts = generate_simulated_trends(patient_inputs, prediction["risk_score"])
    trend_cols = st.columns([1.7, 1])
    with trend_cols[0]:
        marker_options = ["CA19-9", "Glucose/HbA1c", "Weight", "LYVE1", "REG1B", "TFF1"]
        selected_markers = st.multiselect(
            "Markers to display",
            marker_options,
            default=["CA19-9", "Glucose/HbA1c", "Weight"],
            key="trend_marker_selection",
        )
        if not selected_markers:
            selected_markers = ["CA19-9"]
        melted = trend_df.melt(id_vars="month", var_name="marker", value_name="value")
        filtered = melted[melted["marker"].isin(selected_markers)]
        fig = px.line(
            filtered,
            x="month",
            y="value",
            color="marker",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#7fd7ff", "#65d8c6", "#ffd67a", "#ff8e8e", "#85a8ff", "#d79cff"],
            title="Simulated 6-month trajectories",
        )
        apply_plot_style(fig, height=460)
        fig.update_layout(legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with trend_cols[1]:
        st.markdown(
            """
            <div class="section-card">
                <div class="eyebrow">Trend logic</div>
                <h4>What is being monitored</h4>
                <p>These simulated trajectories are meant to show how multiple weak signals can become more concerning when they rise together over time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Trend alerts")
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("No major simulated trend alerts triggered under the current input profile.")


def render_model_performance(trained_bundle: dict[str, Any]) -> None:
    st.subheader("Model Performance")
    st.caption("Metrics below are measured on a stratified holdout test set. Sensitivity and specificity use a 0.50 threshold.")

    comparison_rows = []
    for model_name, metric_values in trained_bundle["metrics"].items():
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metric_values["accuracy"], 3),
                "ROC-AUC": round(metric_values["roc_auc"], 3),
                "Sensitivity": round(metric_values["sensitivity"], 3),
                "Specificity": round(metric_values["specificity"], 3),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("ROC-AUC", ascending=False)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    perf_cols = st.columns(2)
    with perf_cols[0]:
        roc_fig = go.Figure()
        for model_name, metric_values in trained_bundle["metrics"].items():
            roc_fig.add_trace(
                go.Scatter(
                    x=metric_values["fpr"],
                    y=metric_values["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC {metric_values['roc_auc']:.3f})",
                    line=dict(width=3),
                )
            )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(color="#6c7a89", dash="dash"),
            )
        )
        roc_fig.update_layout(
            title="ROC curve comparison",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        apply_plot_style(roc_fig, height=420)
        st.plotly_chart(roc_fig, use_container_width=True)

    with perf_cols[1]:
        best_model_name = trained_bundle["best_model_name"]
        cm = trained_bundle["metrics"][best_model_name]["confusion_matrix"]
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Teal",
            labels=dict(x="Predicted label", y="Actual label", color="Count"),
            x=["Non-PDAC", "PDAC"],
            y=["Non-PDAC", "PDAC"],
            title=f"Confusion matrix: {best_model_name}",
        )
        apply_plot_style(cm_fig, height=420)
        st.plotly_chart(cm_fig, use_container_width=True)


def render_explainability(
    prepared: PreparedData,
    trained_bundle: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    st.subheader("Explainability")
    st.caption("PancreaTrack prioritizes interpretable risk signals. SHAP is used when available; otherwise feature importance is shown.")

    model_name = trained_bundle["best_model_name"]
    pipeline = trained_bundle["best_model"]
    shap_result = calculate_shap_values(
        model_name=model_name,
        pipeline=pipeline,
        x_reference=trained_bundle["x_train"],
        patient_frame=build_patient_frame(st.session_state["patient_inputs"]),
    )

    if shap_result:
        summary_frame = pd.DataFrame(
            {
                "feature": shap_result["feature_names"],
                "importance": shap_result["summary_values"],
            }
        )
        summary_frame["display_feature"] = summary_frame["feature"].map(
            lambda val: DISPLAY_NAMES.get(val, val)
        )
        summary_frame = summary_frame.sort_values("importance", ascending=False).head(10)
        title = "Mean absolute SHAP values"
    else:
        classifier = pipeline.named_steps["classifier"]
        if model_name == "Logistic Regression":
            importance_values = np.abs(classifier.coef_[0])
        else:
            importance_values = classifier.feature_importances_
        summary_frame = pd.DataFrame(
            {
                "feature": get_feature_names(pipeline),
                "importance": importance_values,
            }
        )
        summary_frame["display_feature"] = summary_frame["feature"].map(
            lambda val: DISPLAY_NAMES.get(val, val)
        )
        summary_frame = summary_frame.sort_values("importance", ascending=False).head(10)
        title = "Fallback feature importance"

    summary_fig = px.bar(
        summary_frame.sort_values("importance"),
        x="importance",
        y="display_feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Tealgrn",
        title=title,
    )
    apply_plot_style(summary_fig, height=420)
    summary_fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(summary_fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="info-card">
            <h4>Patient-level explanation</h4>
            <p>{prediction['narrative']}</p>
            <p class="small-note">
                Model interpretation is intended to support discussion around weak signal aggregation,
                not to replace imaging, pathology, or specialist review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    distribution_fig = make_subplots(rows=1, cols=2, subplot_titles=("LYVE1 by diagnosis", "TFF1 by diagnosis"))
    for idx, marker in enumerate(["LYVE1", "TFF1"], start=1):
        for label, color in [("Control", "#59d0ff"), ("Benign", "#ffd166"), ("PDAC", "#ff6b6b")]:
            group_value = {"Control": 0, "Benign": 1, "PDAC": 2}[label]
            distribution_fig.add_trace(
                go.Box(
                    y=prepared.dataframe.loc[prepared.dataframe["diagnosis"] == group_value, marker],
                    name=label,
                    marker_color=color,
                    boxmean=True,
                    showlegend=(idx == 1),
                ),
                row=1,
                col=idx,
            )
    apply_plot_style(distribution_fig, height=500)
    st.plotly_chart(distribution_fig, use_container_width=True)


def collect_sidebar_inputs(prepared: PreparedData) -> dict[str, Any]:
    biomarker_defaults = {
        "creatinine": float(prepared.dataframe["creatinine"].median()),
        "LYVE1": float(prepared.dataframe["LYVE1"].median()),
        "REG1B": float(prepared.dataframe["REG1B"].median()),
        "TFF1": float(prepared.dataframe["TFF1"].median()),
        "plasma_CA19_9": float(prepared.dataframe["plasma_CA19_9"].median()),
        "age": int(prepared.dataframe["age"].median()),
    }
    with st.sidebar:
        st.header("Patient Inputs")
        st.caption("Use these fields to explore how combined signals shift the educational risk estimate.")
        st.markdown('<div class="eyebrow">Demographics</div>', unsafe_allow_html=True)
        age = st.slider("Age", 25, 95, biomarker_defaults["age"])
        sex = st.selectbox("Sex", ["F", "M"], index=0)
        st.markdown('<div class="eyebrow">Urinary biomarkers</div>', unsafe_allow_html=True)
        creatinine = st.number_input("Creatinine", min_value=0.0, value=biomarker_defaults["creatinine"], step=0.1)
        lyve1 = st.number_input("LYVE1", min_value=0.0, value=biomarker_defaults["LYVE1"], step=0.1)
        reg1b = st.number_input("REG1B", min_value=0.0, value=biomarker_defaults["REG1B"], step=1.0)
        tff1 = st.number_input("TFF1", min_value=0.0, value=biomarker_defaults["TFF1"], step=1.0)
        st.markdown('<div class="eyebrow">Metabolic and symptom signals</div>', unsafe_allow_html=True)
        ca19_9 = st.number_input("CA19-9", min_value=0.0, value=biomarker_defaults["plasma_CA19_9"], step=1.0)
        glucose = st.number_input("Glucose / HbA1c", min_value=0.0, value=102.0, step=1.0)
        weight_loss_pct = st.slider("Weight loss percentage", 0.0, 20.0, 3.0, 0.5)
        new_onset_diabetes = st.toggle("New-onset diabetes", value=False)
        pain = st.toggle("Abdominal or back pain", value=False)
        jaundice = st.toggle("Jaundice", value=False)
        fatigue = st.toggle("Fatigue", value=False)

    patient_inputs = {
        "age": age,
        "sex": sex,
        "creatinine": float(creatinine),
        "LYVE1": float(lyve1),
        "REG1B": float(reg1b),
        "TFF1": float(tff1),
        "CA19_9": float(ca19_9),
        "glucose_or_hba1c": float(glucose),
        "weight_loss_pct": float(weight_loss_pct),
        "new_onset_diabetes": bool(new_onset_diabetes),
        "pain": bool(pain),
        "jaundice": bool(jaundice),
        "fatigue": bool(fatigue),
    }
    st.session_state["patient_inputs"] = patient_inputs
    return patient_inputs


def main() -> None:
    inject_theme()
    st.title("PancreaTrack")
    st.caption("Early-risk pancreatic cancer dashboard for educational clinical decision support.")

    raw_df = load_data()
    if raw_df is None:
        st.info("Add the Kaggle CSV at `data/archive/Debernardi et al 2020 data.csv` or upload it above.")
        return

    prepared = clean_data(raw_df)
    trained_bundle = train_models(prepared.dataframe)
    patient_inputs = collect_sidebar_inputs(prepared)
    prediction = predict_patient_risk(patient_inputs, trained_bundle)

    tabs = st.tabs(
        [
            "Overview",
            "Patient Dashboard",
            "Biomarker Trends",
            "Model Performance",
            "Explainability",
        ]
    )

    with tabs[0]:
        render_overview(prepared, trained_bundle)
    with tabs[1]:
        render_patient_dashboard(patient_inputs, prediction)
    with tabs[2]:
        render_biomarker_trends(patient_inputs, prediction)
    with tabs[3]:
        render_model_performance(trained_bundle)
    with tabs[4]:
        render_explainability(prepared, trained_bundle, prediction)


if __name__ == "__main__":
    main()
