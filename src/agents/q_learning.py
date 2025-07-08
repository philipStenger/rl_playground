"""
Q-Learning agent implementation.
"""

import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    """Q-Learning agent for discrete state and action spaces."""
    
    def __init__(self, 
                 env,
                 lr: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table using defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # For continuous state spaces, we need to discretize
        self.state_bins = self._create_bins()
    
    def _create_bins(self, num_bins: int = 20):
        """Create bins for discretizing continuous state spaces."""
        if hasattr(self.env.observation_space, 'high'):
            bins = []
            for i in range(self.env.observation_space.shape[0]):
                low = self.env.observation_space.low[i]
                high = self.env.observation_space.high[i]
                
                # Handle infinite bounds
                if np.isinf(low):
                    low = -10
                if np.isinf(high):
                    high = 10
                
                bins.append(np.linspace(low, high, num_bins))
            return bins
        return None
    
    def _discretize_state(self, state):
        """Discretize continuous state into discrete bins."""
        if self.state_bins is None:
            return tuple(state)
        
        discrete_state = []
        for i, value in enumerate(state):
            bin_index = np.digitize(value, self.state_bins[i]) - 1
            bin_index = np.clip(bin_index, 0, len(self.state_bins[i]) - 2)
            discrete_state.append(bin_index)
        
        return tuple(discrete_state)
    
    def act(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy."""
        discrete_state = self._discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Q-learning update
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[discrete_next_state])
        
        # Update Q-value
        self.q_table[discrete_state][action] += self.lr * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save Q-table and parameters."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'gamma': self.gamma,
            'state_bins': self.state_bins
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load Q-table and parameters."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), data['q_table'])
        self.epsilon = data['epsilon']
        self.lr = data['lr']
        self.gamma = data['gamma']
        self.state_bins = data['state_bins']
