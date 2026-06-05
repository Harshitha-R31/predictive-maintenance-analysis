import pandas as pd
import numpy as np
import streamlit as st

def load_and_clean(file):
    """Load CSV and clean the data"""
    df = pd.read_csv(file)

    # ── Debug: show actual columns ─────────────────────────────────────────
    st.write("📋 Columns found in your CSV:", list(df.columns))

    # ── Auto-detect numeric columns ────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove known non-sensor columns
    exclude = ['Failure_Status', 'failure', 'Machine failure', 'target',
               'Target', 'label', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    sensor_cols = [c for c in numeric_cols if c not in exclude]

    # ── Map to standard names by keyword matching ──────────────────────────
    def find_col(keywords, cols):
        for kw in keywords:
            for c in cols:
                if kw.lower() in c.lower():
                    return c
        return None

    temp_col  = find_col(['temp', 'temperature'], sensor_cols)
    vib_col   = find_col(['vibrat', 'rotational', 'rpm', 'torque', 'speed'], sensor_cols)
    volt_col  = find_col(['volt', 'tool', 'wear', 'power', 'current'], sensor_cols)
    fail_col  = find_col(['failure', 'fail', 'target', 'label'], df.columns.tolist())
    id_col    = find_col(['id', 'equipment', 'machine', 'product', 'udi'], df.columns.tolist())
    time_col  = find_col(['time', 'timestamp', 'date'], df.columns.tolist())

    # If still not found, assign first 3 numeric cols
    if temp_col is None and len(sensor_cols) > 0:
        temp_col = sensor_cols[0]
    if vib_col is None and len(sensor_cols) > 1:
        vib_col = sensor_cols[1]
    if volt_col is None and len(sensor_cols) > 2:
        volt_col = sensor_cols[2]

    st.write(f"✅ Mapped → Temperature: `{temp_col}` | Vibration: `{vib_col}` | Voltage: `{volt_col}` | Failure: `{fail_col}`")

    # ── Build standardized dataframe ───────────────────────────────────────
    std = pd.DataFrame()

    std['Timestamp']    = pd.to_datetime(df[time_col], errors='coerce') if time_col else pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
    std['Equipment_ID'] = df[id_col].astype(str) if id_col else 'EQ001'
    std['Temperature_C']  = pd.to_numeric(df[temp_col], errors='coerce') if temp_col else 0
    std['Vibration_ms2']  = pd.to_numeric(df[vib_col],  errors='coerce') if vib_col  else 0
    std['Voltage_V']      = pd.to_numeric(df[volt_col], errors='coerce') if volt_col else 0

    if fail_col:
        std['Failure_Status'] = pd.to_numeric(df[fail_col], errors='coerce').fillna(0).astype(int)

    std = std.sort_values('Timestamp').reset_index(drop=True)
    std = std.fillna(std.median(numeric_only=True))

    return std


def add_features(df):
    """Add rolling stats and Z-score features"""

    for col, out in [('Vibration_ms2', 'Vibration_RollingMean'),
                     ('Temperature_C', 'Temp_RollingMean'),
                     ('Voltage_V',     'Voltage_RollingMean')]:
        df[out] = df[col].rolling(10).mean().fillna(df[col])

    for col in ['Temperature_C', 'Vibration_ms2', 'Voltage_V']:
        mean = df[col].mean()
        std  = df[col].std()
        df[f'{col}_Zscore'] = (df[col] - mean) / (std if std != 0 else 1)

    return df


def flag_anomalies(df, threshold=2.5):
    """Flag rows where any sensor Z-score exceeds threshold"""
    df['Anomaly_Alert'] = np.where(
        (df['Temperature_C_Zscore'].abs() > threshold) |
        (df['Vibration_ms2_Zscore'].abs() > threshold) |
        (df['Voltage_V_Zscore'].abs()     > threshold),
        1, 0
    )
    return df


def get_feature_columns():
    return [
        'Temperature_C', 'Vibration_ms2', 'Voltage_V',
        'Vibration_RollingMean', 'Temp_RollingMean', 'Voltage_RollingMean',
        'Temperature_C_Zscore', 'Vibration_ms2_Zscore', 'Voltage_V_Zscore'
    ]
