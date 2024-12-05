# src/data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.power_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load power and weather data"""
        try:
            # Load datasets
            power_data = pd.read_csv(self.config['power_data_path'])
            weather_data = pd.read_csv(self.config['weather_data_path'])

            # Convert timestamps
            power_data['StartDate'] = pd.to_datetime(power_data['StartDate'])
            weather_data['Date'] = pd.to_datetime(weather_data['Date'])

            self.logger.info(f"Power data shape: {power_data.shape}")
            self.logger.info(f"Weather data shape: {weather_data.shape}")

            return power_data, weather_data

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def merge_data(self, power_data: pd.DataFrame, 
                  weather_data: pd.DataFrame) -> pd.DataFrame:
        """Merge power and weather data"""
        try:
            # Merge on date
            merged_df = pd.merge(
                power_data,
                weather_data,
                left_on=power_data['StartDate'].dt.date,
                right_on=weather_data['Date'].dt.date,
                how='inner'
            )

            self.logger.info(f"Merged data shape: {merged_df.shape}")
            return merged_df

        except Exception as e:
            self.logger.error(f"Error merging data: {str(e)}")
            raise

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        try:
            # Scale target variable
            target = data[self.config['target']].values.reshape(-1, 1)
            scaled_target = self.power_scaler.fit_transform(target)

            # Scale features
            features = data[self.config['weather_features']].values
            scaled_features = self.feature_scaler.fit_transform(features)

            return scaled_features, scaled_target

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def create_dataloaders(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders"""
        try:
            # Calculate split indices
            train_size = int((1 - self.config['train_test_split']) * len(X))
            val_size = int(self.config['validation_split'] * train_size)
            train_size = train_size - val_size

            # Split data
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            test_dataset = TimeSeriesDataset(X_test, y_test)

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size'],
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config['batch_size'],
                shuffle=False
            )

            return train_loader, val_loader, test_loader

        except Exception as e:
            self.logger.error(f"Error creating dataloaders: {str(e)}")
            raise