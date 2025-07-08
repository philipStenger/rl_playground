"""
REINFORCE (Policy Gradient) agent implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = [128, 128]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    """REINFORCE (Policy Gradient) agent."""
    
    def __init__(self,
                 env,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.device = device
        
        # Policy network
        self.policy_net = PolicyNetwork(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
    
    def act(self, state, training: bool = True):
        """Choose action according to policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_net(state_tensor)
            dist = Categorical(action_probs)
            
            if training:
                action = dist.sample()
                self.log_probs.append(dist.log_prob(action))
                return action.item()
            else:
                # For evaluation, choose action with highest probability
                return action_probs.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store reward for current episode."""
        self.rewards.append(reward)
    
    def learn(self, state=None, action=None, reward=None, next_state=None, done=None):
        """Update policy using REINFORCE algorithm."""
        if not done:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
    
    def save(self, filepath: str):
        """Save policy network."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load policy network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
