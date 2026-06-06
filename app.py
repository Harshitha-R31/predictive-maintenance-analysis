import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

from data_pipeline import load_and_clean, add_features, flag_anomalies, get_feature_columns
from model         import train_model, evaluate_model, get_shap_values, plot_shap, load_model

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equipment Failure Dashboard",
    layout="wide",
    page_icon="⚙️"
)

st.title("⚙️ Predictive Equipment Failure Dashboard")
st.caption("Industry 4.0 · Real-time Anomaly Detection & ML Failure Prediction")

st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #EBF5FB;
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #D6EAF8;
        }

        /* Card/metric background */
        [data-testid="stMetric"] {
            background-color: #D6EAF8;
            border-radius: 10px;
            padding: 10px;
        }

        /* Text color */
        html, body, [class*="css"] {
            color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("📂 Upload Sensor CSV", type="csv")
    st.divider()
    threshold = st.slider("🔔 Anomaly Z-score Threshold", 1.5, 4.0, 2.5, 0.1,
                          help="Lower = more sensitive alerts")
    st.divider()
    train_new = st.checkbox("🤖 Retrain ML Model", value=True,
                            help="Uncheck to use saved model if available")

# ── Main Logic ────────────────────────────────────────────────────────────────
if uploaded:

    # 1. Load & process
    df = load_and_clean(uploaded)
    df = add_features(df)
    df = flag_anomalies(df, threshold)

    # 2. Equipment filter
    with st.sidebar:
        st.divider()
        equip_ids = ["All"] + sorted(df['Equipment_ID'].unique().tolist())
        selected  = st.selectbox("🏭 Filter Equipment ID", equip_ids)

    filtered_df = df if selected == "All" else df[df['Equipment_ID'] == selected]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.subheader("📊 Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records",         len(filtered_df))
    k2.metric("🚨 Anomaly Alerts",     int(filtered_df['Anomaly_Alert'].sum()))
    k3.metric("⚠️ Failures",           int(filtered_df['Failure_Status'].sum())
               if 'Failure_Status' in filtered_df.columns else "N/A")
    k4.metric("🌡️ Avg Temperature",   f"{filtered_df['Temperature_C'].mean():.1f} °C")
    k5.metric("📳 Avg Vibration",      f"{filtered_df['Vibration_ms2'].mean():.3f} m/s²")

    st.divider()

    # ── Sensor Trend Charts ───────────────────────────────────────────────────
    st.subheader("📈 Sensor Trends Over Time")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_temp = px.line(
            filtered_df, x='Timestamp', y='Temperature_C',
            title="Temperature (°C)",
            color_discrete_sequence=['#E05C2A']
        )
        fig_temp.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_b:
        fig_vib = px.line(
            filtered_df, x='Timestamp', y='Vibration_ms2',
            title="Vibration (m/s²)",
            color_discrete_sequence=['#2A7AE0']
        )
        fig_vib.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_vib, use_container_width=True)

    # Voltage chart
    fig_volt = px.line(
        filtered_df, x='Timestamp', y='Voltage_V',
        title="Voltage (V)",
        color_discrete_sequence=['#2ABA7A']
    )
    fig_volt.update_layout(height=250, margin=dict(t=40, b=20))
    st.plotly_chart(fig_volt, use_container_width=True)

    st.divider()

    # ── Anomaly Alerts Table ──────────────────────────────────────────────────
    st.subheader("🚨 Anomaly Alerts")
    alerts = filtered_df[filtered_df['Anomaly_Alert'] == 1]

    if len(alerts) > 0:
        st.error(f"⚠️ {len(alerts)} anomalies detected in sensor readings!")
        st.dataframe(
            alerts[['Timestamp', 'Equipment_ID',
                     'Temperature_C', 'Vibration_ms2', 'Voltage_V',
                     'Temperature_C_Zscore', 'Vibration_ms2_Zscore']].round(3),
            use_container_width=True
        )
    else:
        st.success("✅ No anomalies detected with current threshold.")

    st.divider()

    # ── ML Model ──────────────────────────────────────────────────────────────
    st.subheader("🤖 Machine Learning — Failure Prediction")

    if 'Failure_Status' in df.columns:

        features = get_feature_columns()
        X_full   = df[features].fillna(0)

        # Train or load model
        os.makedirs("model", exist_ok=True)

        if train_new:
            with st.spinner("Training Random Forest model..."):
                model, X_test, y_test = train_model(df)
            st.success("✅ Model trained successfully!")
        else:
            model = load_model()
            if model is None:
                st.warning("No saved model found. Training new model...")
                model, X_test, y_test = train_model(df)
            else:
                from sklearn.model_selection import train_test_split
                X = df[features].fillna(0)
                y = df['Failure_Status']
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
                st.success("✅ Loaded saved model.")

        # Predict on full dataset
        df['Predicted_Failure'] = model.predict(X_full)
        proba = model.predict_proba(X_full)
        if proba.shape[1] == 1:
            classes = model.classes_
            df['Failure_Probability'] = proba[:, 0] if classes[0] == 1 else 0.0
            st.warning('⚠️ Model trained on only one class. Dataset may have no failures.')
        else:
            df['Failure_Probability'] = proba[:, 1]

        # Model metrics
        from sklearn.metrics import accuracy_score, f1_score
        y_pred_test = model.predict(X_test)

        m1, m2, m3 = st.columns(3)
        m1.metric("Model Accuracy",
                  f"{accuracy_score(y_test, y_pred_test):.2%}")
        m2.metric("F1 Score",
                  f"{f1_score(y_test, y_pred_test):.2f}")
        m3.metric("High Risk Machines",
                  int((df['Failure_Probability'] > 0.7).sum()))

        st.divider()

        # Failure probability chart
        st.subheader("📉 Failure Probability Over Time")
        fig_prob = px.line(
            df, x='Timestamp', y='Failure_Probability',
            color_discrete_sequence=['#D62728'],
            title="ML Predicted Failure Probability"
        )
        fig_prob.add_hline(y=0.7, line_dash="dash",
                           line_color="orange",
                           annotation_text="High Risk Threshold (0.7)")
        fig_prob.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_prob, use_container_width=True)

        # High-risk machines table
        high_risk = df[df['Failure_Probability'] > 0.7][
            ['Timestamp', 'Equipment_ID',
             'Temperature_C', 'Vibration_ms2',
             'Failure_Probability']
        ].sort_values('Failure_Probability', ascending=False)

        if len(high_risk) > 0:
            st.error(f"🔴 {len(high_risk)} high-risk readings (probability > 70%)")
            st.dataframe(high_risk.round(3), use_container_width=True)

        st.divider()

        # ── SHAP Explainability ───────────────────────────────────────────────
        st.subheader("🔍 Explainability — Why is the model predicting failure?")

        with st.spinner("Computing SHAP values..."):
            explainer, shap_values = get_shap_values(model, X_test)
            fig_shap = plot_shap(shap_values, X_test)

        st.pyplot(fig_shap)
        st.caption("This chart shows which sensor features have the most influence on the failure prediction. Longer bars = higher impact.")

    else:
        st.warning("'Failure_Status' column not found in dataset. ML model requires labeled data.")

else:
    # Welcome screen
    st.info("👈 Upload a sensor CSV file from the sidebar to get started.")
    st.markdown("""
    ### 📋 Expected CSV Format
    Your CSV should have these columns:

    | Column | Type | Example |
    |---|---|---|
    | `Timestamp` | datetime | 2024-01-01 10:00:00 |
    | `Equipment_ID` | string | EQ001 |
    | `Temperature_C` | float | 72.5 |
    | `Vibration_ms2` | float | 0.84 |
    | `Voltage_V` | float | 220.3 |
    | `Failure_Status` | int (0/1) | 0 |

    **Dataset:** Download the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) from Kaggle.
    """)
