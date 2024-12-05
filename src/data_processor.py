# src/data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.power_scaler = MinMaxScaler()
        self.weather_scaler = MinMaxScaler()
        
    def load_data(self):
        # Load power usage data
        power_data = pd.read_csv('data/power_usage_2016_to_2020.csv')
        weather_data = pd.read_csv('data/weather_2016_2020_daily.csv')
        
        # Clean and standardize date formats
        power_data['StartDate'] = pd.to_datetime(
            power_data['StartDate'], 
            format='%Y-%m-%d %H-%M-%S',
            errors='coerce'
        )
        
        weather_data['Date'] = pd.to_datetime(
            weather_data['Date'],
            format='%Y-%m-%d',
            errors='coerce'
        )
        
        # Drop rows with invalid dates
        power_data = power_data.dropna(subset=['StartDate'])
        weather_data = weather_data.dropna(subset=['Date'])
        
        return power_data, weather_data
    
    def merge_data(self, power_data, weather_data):
        # Extract date from power data for merging
        power_data['Date'] = power_data['StartDate'].dt.date
        weather_data['Date'] = weather_data['Date'].dt.date
        
        # Merge the datasets
        merged_data = pd.merge(power_data, weather_data, 
                             on='Date', how='left')
        
        return merged_data