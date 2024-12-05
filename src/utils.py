#src/utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'results/models', 'results/plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def plot_power_consumption(data, title, save_path=None):
    """Plot power consumption over time"""
    plt.figure(figsize=(15, 6))
    plt.plot(data['StartDate'], data['Watts'])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Power Consumption (Watts)')
    if save_path:
        plt.savefig(save_path)
    plt.close()