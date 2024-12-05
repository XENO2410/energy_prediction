#src/config.py
CONFIG = {
    # Data parameters
    'power_data_path': 'data/power_usage_2016_to_2020.csv',
    'weather_data_path': 'data/weather_2016_2020_daily.csv',
    
    # Feature engineering parameters
    'sequence_length': 24,  # 24 hours of data
    'lag_features': [1, 7, 14, 30],
    'rolling_windows': [6, 12, 24],
    
    # Model parameters
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    
    # Training parameters
    'train_test_split': 0.2,
    'validation_split': 0.1,
    'random_seed': 42,
    
    # Features to use
    'weather_features': [
        'Temp_avg', 'Hum_avg', 'Wind_avg', 'Precipit',
        'Press_avg', 'Temp_min', 'Temp_max'
    ],
    
    # Target variable
    'target': 'Value (kWh)'
}