#src/config.py
CONFIG = {
    'random_seed': 42,
    'train_test_split': 0.2,
    'validation_split': 0.1,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'sequence_length': 24,  # 24 hours for daily prediction
    'feature_columns': [
        'Value (kWh)',
        'Temp_avg',
        'Hum_avg',
        'Wind_avg',
        'Press_avg'
    ]
}