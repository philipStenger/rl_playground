# Reinforcement Learning Playground

A playground for experimenting with reinforcement learning algorithms and environments.

## Overview

This project provides implementations of various RL algorithms including:
- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient methods (REINFORCE)
- Actor-Critic methods
- Proximal Policy Optimization (PPO)

## Features

- Classic control environments (CartPole, MountainCar, etc.)
- Custom grid world environments
- Visualization tools for training progress
- Hyperparameter tuning utilities
- Model saving and loading

## Setup

1. Create a virtual environment:
```bash
python -m venv env
```

2. Activate the virtual environment:
```bash
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Train different RL algorithms:
```bash
# Q-Learning on CartPole
python main.py --algorithm q_learning --env CartPole-v1

# DQN on CartPole
python main.py --algorithm dqn --env CartPole-v1

# Policy Gradient on CartPole
python main.py --algorithm reinforce --env CartPole-v1
```

## Project Structure

- `src/` - Main source code
- `agents/` - RL agent implementations
- `environments/` - Environment wrappers and custom environments
- `utils/` - Utility functions and visualization tools
- `experiments/` - Experiment configurations and results
- `models/` - Saved model checkpoints
