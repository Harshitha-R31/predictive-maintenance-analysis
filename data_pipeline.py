import pandas as pd
import numpy as np

def load_and_clean(file):
    """Load CSV and clean the data"""
    df = pd.read_csv(file)

    # ── Rename columns to standard names ──────────────────────────────────
    # Handles different column name formats from various datasets
    rename_map = {
        # Timestamp variants
        'timestamp': 'Timestamp', 'time': 'Timestamp', 'date': 'Timestamp',
        'Date': 'Timestamp', 'Time': 'Timestamp',

        # Equipment ID variants
        'equipment_id': 'Equipment_ID', 'machine_id': 'Equipment_ID',
        'Machine_ID': 'Equipment_ID', 'UDI': 'Equipment_ID',
        'Product ID': 'Equipment_ID', 'Product_ID': 'Equipment_ID',

        # Temperature variants
        'temperature': 'Temperature_C', 'temp': 'Temperature_C',
        'Temperature': 'Temperature_C', 'Air temperature [K]': 'Temperature_C',
        'Process temperature [K]': 'Temperature_C',
        'air_temperature': 'Temperature_C',

        # Vibration variants
        'vibration': 'Vibration_ms2', 'Vibration': 'Vibration_ms2',
        'Rotational speed [rpm]': 'Vibration_ms2',
        'rotational_speed': 'Vibration_ms2', 'rpm': 'Vibration_ms2',
        'Torque [Nm]': 'Vibration_ms2',

        # Voltage variants
        'voltage': 'Voltage_V', 'Voltage': 'Voltage_V',
        'Tool wear [min]': 'Voltage_V', 'tool_wear': 'Voltage_V',
        'power': 'Voltage_V', 'Power': 'Voltage_V',

        # Failure variants
        'failure': 'Failure_Status', 'Failure': 'Failure_Status',
        'Machine failure': 'Failure_Status', 'target': 'Failure_Status',
        'Target': 'Failure_Status', 'label': 'Failure_Status',
        'failure_status': 'Failure_Status',
    }
    df = df.rename(columns=rename_map)

    # If Timestamp column missing, create a dummy one
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')

    # If Equipment_ID column missing, create a dummy one
    if 'Equipment_ID' not in df.columns:
        df['Equipment_ID'] = 'EQ001'

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    return df


def add_features(df):
    """Add rolling stats and Z-score features"""
    
    # Rolling mean and std (window of 10 readings)
    df['Vibration_RollingMean'] = df['Vibration_ms2'].rolling(10).mean().fillna(df['Vibration_ms2'])
    df['Temp_RollingMean']      = df['Temperature_C'].rolling(10).mean().fillna(df['Temperature_C'])
    df['Voltage_RollingMean']   = df['Voltage_V'].rolling(10).mean().fillna(df['Voltage_V'])

    # Z-score for each sensor column
    for col in ['Temperature_C', 'Vibration_ms2', 'Voltage_V']:
        mean = df[col].mean()
        std  = df[col].std()
        df[f'{col}_Zscore'] = (df[col] - mean) / std

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
    """Return list of feature columns used for ML"""
    return [
        'Temperature_C', 'Vibration_ms2', 'Voltage_V',
        'Vibration_RollingMean', 'Temp_RollingMean', 'Voltage_RollingMean',
        'Temperature_C_Zscore', 'Vibration_ms2_Zscore', 'Voltage_V_Zscore'
    ]
