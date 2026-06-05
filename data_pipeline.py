import pandas as pd
import numpy as np

def load_and_clean(file):
    """Load CSV and clean the data"""
    df = pd.read_csv(file)
    
    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
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
