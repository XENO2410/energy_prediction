#src/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        # Update column names to match your dataset
        self.power_column = 'Value (kWh)'
        self.temp_column = 'Temp_avg'
        self.humidity_column = 'Hum_avg'
        self.wind_column = 'Wind_avg'
        
    def create_features(self, df):
        """Create time-based features and engineer additional features"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['StartDate'].dt.hour
        df['day'] = df['StartDate'].dt.day
        df['month'] = df['StartDate'].dt.month
        
        # Use existing day_of_week column or create if needed
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['StartDate'].dt.dayofweek
            
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate rolling statistics for power consumption
        df['rolling_mean_24h'] = df[self.power_column].rolling(window=24).mean()
        df['rolling_std_24h'] = df[self.power_column].rolling(window=24).std()
        
        # Create lag features
        for i in [1, 2, 3, 24]:  # 1, 2, 3 hours and 1 day lag
            df[f'lag_{i}h'] = df[self.power_column].shift(i)
        
        # Add weather-based features
        df['temp_humidity_interaction'] = df[self.temp_column] * df[self.humidity_column]
        
        # Drop rows with NaN values created by rolling and lag features
        df = df.dropna()
        
        return df
    
    def scale_features(self, df, is_training=True):
        """Scale numerical features"""
        features_to_scale = [
            self.power_column,
            self.temp_column,
            self.humidity_column,
            self.wind_column,
            'rolling_mean_24h',
            'rolling_std_24h',
            'temp_humidity_interaction'
        ]
        
        if is_training:
            for feature in features_to_scale:
                if feature in df.columns:
                    self.scalers[feature] = MinMaxScaler()
                    df[feature] = self.scalers[feature].fit_transform(
                        df[feature].values.reshape(-1, 1)
                    )
        else:
            for feature in features_to_scale:
                if feature in df.columns and feature in self.scalers:
                    df[feature] = self.scalers[feature].transform(
                        df[feature].values.reshape(-1, 1)
                    )
        
        return df
    
    def prepare_sequences(self, df):
        """Prepare sequences for time series prediction"""
        sequence_length = self.config['sequence_length']
        
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:(i + sequence_length)]
            target = df.iloc[i + sequence_length]['Watts']
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)