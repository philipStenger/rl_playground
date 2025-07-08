"""
Environment wrappers and utilities.
"""

import gymnasium as gym
import numpy as np

def make_env(env_name: str, seed: int = None):
    """Create and configure environment."""
    env = gym.make(env_name)
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    
    return env

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to have zero mean and unit variance."""
    
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape[0])
        self.obs_var = np.ones(env.observation_space.shape[0])
        self.obs_count = 0
    
    def observation(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        self.obs_var += delta * (obs - self.obs_mean)
        
        # Normalize
        return (obs - self.obs_mean) / np.sqrt(self.obs_var / self.obs_count + self.epsilon)
