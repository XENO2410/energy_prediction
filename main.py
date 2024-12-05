#main.py
from src.config import CONFIG
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.models import LSTMModel
from src.train import Trainer
from src.utils import setup_logging, create_directories
import logging

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
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()