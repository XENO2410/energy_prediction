#src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import logging
from tqdm import tqdm
from .utils import plot_training_history, save_model

class Trainer:
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate']
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y.view(-1, 1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y.view(-1, 1))
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(self.model, 'results/models/best_model.pth')
            
            self.logger.info(
                f'Epoch {epoch+1}/{self.config["epochs"]} - '
                f'Train Loss: {train_loss:.4f} - '
                f'Val Loss: {val_loss:.4f}'
            )
        
        # Plot training history
        plot_training_history(
            train_losses, 
            val_losses, 
            'results/plots/training_history.png'
        )
        
        return train_losses, val_losses