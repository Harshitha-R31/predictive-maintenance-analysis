import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (classification_report,
                                      confusion_matrix,
                                      accuracy_score)
from data_pipeline import get_feature_columns


# ── Train & Save ──────────────────────────────────────────────────────────────

def train_model(df):
    """Train Random Forest classifier and return model + test split"""
    features = get_feature_columns()

    X = df[features].fillna(0)
    y = df['Failure_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'   # handles imbalanced failure data
    )
    model.fit(X_train, y_train)

    # Save model to disk
    joblib.dump(model, 'model/model.pkl')
    print("✅ Model saved to model/model.pkl")

    return model, X_test, y_test


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Print accuracy, classification report, confusion matrix"""
    y_pred = model.predict(X_test)

    print(f"\n📊 Accuracy : {accuracy_score(y_test, y_pred):.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['Normal', 'Failure']))
    print("🔲 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


# ── SHAP Explainability ───────────────────────────────────────────────────────

def get_shap_values(model, X_test):
    """Compute SHAP values for explainability"""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_shap(shap_values, X_test):
    """Return SHAP summary bar chart as matplotlib figure"""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Handle both old and new SHAP output formats
    # Old SHAP: shap_values is a list [class0_array, class1_array]
    # New SHAP: shap_values is a 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        # Old format — pick class 1 (Failure)
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # New format — slice class 1
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    shap.summary_plot(
        sv, X_test,
        plot_type="bar",
        show=False,
        color="#E05C2A"
    )
    plt.title("Feature Importance (SHAP) — Failure Prediction", fontsize=12)
    plt.tight_layout()
    return fig


# ── Load Saved Model ──────────────────────────────────────────────────────────

def load_model():
    """Load saved model from disk"""
    try:
        model = joblib.load('model/model.pkl')
        return model
    except FileNotFoundError:
        return None


# ── Predict Single Row ────────────────────────────────────────────────────────

def predict_single(model, row_df):
    """Predict failure for a single equipment reading"""
    features = get_feature_columns()
    X        = row_df[features].fillna(0)
    prob     = model.predict_proba(X)[0][1]   # probability of failure
    pred     = model.predict(X)[0]
    return pred, prob
