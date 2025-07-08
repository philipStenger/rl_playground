"""
Training utilities for RL agents.
"""

import numpy as np
from typing import Dict, List
from tqdm import tqdm

def train_agent(agent, env, episodes: int = 1000, render: bool = False, logger=None):
    """Train an RL agent."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Choose action
            action = agent.act(state, training=True)
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            
            # Store experience (if agent supports it)
            if hasattr(agent, 'remember'):
                agent.remember(state, action, reward, next_state, done)
            
            # Learn (if agent supports it)
            if hasattr(agent, 'replay'):
                agent.replay()
            elif hasattr(agent, 'learn'):
                agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Log progress
        if logger and episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_average_reward': np.mean(episode_rewards[-100:])
    }
