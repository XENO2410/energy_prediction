#main.py
from src.config import CONFIG
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.models import LSTMModel
from src.train import Trainer
from src.utils import setup_logging, create_directories
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, test_loader, data_processor, logger):
    """Evaluate the model on test data"""
    model.eval()
    predictions = []
    actuals = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            output = model(X)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.numpy())
    
    predictions = np.array(predictions).reshape(-1)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    logger.info(f"Test Results:")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    return predictions, actuals

def plot_results(predictions, actuals, save_path='results/plots/'):
    """Plot prediction results"""
    # Plot predictions vs actuals
    plt.figure(figsize=(15, 6))
    plt.plot(actuals[:100], label='Actual', color='blue', alpha=0.7)
    plt.plot(predictions[:100], label='Predicted', color='red', alpha=0.7)
    plt.title('Power Consumption: Actual vs Predicted (First 100 Time Steps)')
    plt.xlabel('Time Steps')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}predictions.png')
    plt.close()
    
    # Plot scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Scatter Plot')
    plt.grid(True)
    plt.savefig(f'{save_path}scatter_plot.png')
    plt.close()

def main():
    # Setup
    create_directories()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        data_processor = DataProcessor(CONFIG)
        feature_engineer = FeatureEngineer(CONFIG)
        
        # Load and process data
        logger.info("Loading data...")
        power_data, weather_data = data_processor.load_data()
        
        logger.info("Merging data...")
        merged_data = data_processor.merge_data(power_data, weather_data)
        
        logger.info("Engineering features...")
        processed_data = feature_engineer.process_features(merged_data)
        
        # Prepare data for model
        features, target = data_processor.prepare_data(processed_data)
        X, y = feature_engineer.create_sequences(features, target)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = data_processor.create_dataloaders(X, y)
        
        # Initialize and train model
        logger.info("Training model...")
        model = LSTMModel(CONFIG)
        trainer = Trainer(model, CONFIG)
        
        train_losses, val_losses = trainer.train(train_loader, val_loader)
        
        # Evaluate model
        logger.info("Evaluating model...")
        predictions, actuals = evaluate_model(model, test_loader, data_processor, logger)
        
        # Plot results
        logger.info("Plotting results...")
        plot_results(predictions, actuals)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()