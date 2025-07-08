"""
Visualization utilities for RL training.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from pathlib import Path

def plot_training_results(results: Dict, window_size: int = 100, save_path: Optional[Path] = None):
    """Plot training results."""
    episode_rewards = results['episode_rewards']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot episode rewards
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
    
    # Plot moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label=f'{window_size}-Episode Moving Average')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    if 'episode_lengths' in results:
        episode_lengths = results['episode_lengths']
        ax2.plot(episode_lengths, alpha=0.3, color='green', label='Episode Lengths')
        
        # Plot moving average for lengths
        if len(episode_lengths) >= window_size:
            moving_avg_lengths = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(episode_lengths)), moving_avg_lengths,
                    color='orange', linewidth=2, label=f'{window_size}-Episode Moving Average')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Training Progress: Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_q_values(q_values, actions, save_path: Optional[Path] = None):
    """Plot Q-values for different actions."""
    plt.figure(figsize=(10, 6))
    
    for i, action in enumerate(actions):
        plt.plot(q_values[:, i], label=f'Action {action}', alpha=0.7)
    
    plt.xlabel('Training Step')
    plt.ylabel('Q-Value')
    plt.title('Q-Values During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
