#src/utils.py
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def create_directories():
    """Create necessary directories for saving results"""
    directories = ['results/models', 'results/plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true, y_pred, save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Power Consumption')
    plt.xlabel('Time')
    plt.ylabel('Power Consumption (kWh)')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_model(model, path):
    """Save PyTorch model"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load PyTorch model"""
    model.load_state_dict(torch.load(path))
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }