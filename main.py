#main.py
import torch
from src.config import CONFIG
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.models import LSTMPredictor
from src.train import Trainer
from src.utils import create_directories, plot_power_consumption
import pandas as pd
import numpy as np
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        logging.info("Starting the power consumption prediction pipeline")
        
        # Create necessary directories
        create_directories()
        logging.info("Created necessary directories")
        
        # Initialize components
        data_processor = DataProcessor(CONFIG)
        feature_engineer = FeatureEngineer(CONFIG)
        
        # Load and process data
        logging.info("Loading data...")
        power_data, weather_data = data_processor.load_data()
        merged_data = data_processor.merge_data(power_data, weather_data)
        logging.info(f"Loaded data with shape: {merged_data.shape}")
        
        # Create features
        logging.info("Engineering features...")
        processed_data = feature_engineer.create_features(merged_data)
        scaled_data = feature_engineer.scale_features(processed_data)
        logging.info(f"Processed data shape: {scaled_data.shape}")
        
        # Prepare sequences
        logging.info("Preparing sequences...")
        X, y = feature_engineer.prepare_sequences(scaled_data)
        logging.info(f"Created sequences with shape: {X.shape}")
        
        # Split data
        train_size = int(len(X) * (1 - CONFIG['train_test_split']))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )
        
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=CONFIG['batch_size']
        )
        
        # Initialize model
        input_dim = X_train.shape[2]
        logging.info(f"Creating model with input dimension: {input_dim}")
        model = LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            output_dim=1
        )
        
        # Train model
        trainer = Trainer(model, CONFIG)
        logging.info("Starting training...")
        train_losses, val_losses = trainer.train(
            train_loader,
            test_loader,
            CONFIG['epochs']
        )
        
        # Plot results
        logging.info("Generating plots...")
        plot_power_consumption(
            merged_data,
            'Power Consumption Over Time',
            'results/plots/power_consumption.png'
        )
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()