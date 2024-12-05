#src/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict
import logging

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df['StartDate'].dt.hour
        df['day'] = df['StartDate'].dt.day
        df['month'] = df['StartDate'].dt.month
        df['day_of_week'] = df['StartDate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        df = df.copy()
        
        for lag in self.config['lag_features']:
            df[f'lag_{lag}'] = df[self.config['target']].shift(lag)
            
        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        
        for window in self.config['rolling_windows']:
            df[f'rolling_mean_{window}h'] = df[self.config['target']].rolling(
                window=window).mean()
            df[f'rolling_std_{window}h'] = df[self.config['target']].rolling(
                window=window).std()
            
        return df

    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        seq_length = self.config['sequence_length']
        
        for i in range(len(features) - seq_length):
            X.append(features[i:(i + seq_length)])
            y.append(target[i + seq_length])
            
        return np.array(X), np.array(y)

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        try:
            # Create all features
            df = self.create_time_features(df)
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            
            # Handle missing values
            df = df.dropna()
            
            self.logger.info(f"Feature engineering completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise