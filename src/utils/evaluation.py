"""
Evaluation utilities for RL agents.
"""

import numpy as np

def evaluate_agent(agent, env, episodes: int = 100, render: bool = False):
    """Evaluate an RL agent."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Choose action (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            state, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
