"""
Reinforcement Learning Playground

Main entry point for running RL experiments.
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.reinforce import REINFORCEAgent
from src.environments.wrappers import make_env
from src.utils.training import train_agent
from src.utils.evaluation import evaluate_agent
from src.utils.visualization import plot_training_results
from src.utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning Playground')
    parser.add_argument('--algorithm', type=str, default='dqn',
                       choices=['q_learning', 'dqn', 'reinforce', 'actor_critic', 'ppo'],
                       help='RL algorithm to use')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment to train on')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial epsilon for epsilon-greedy exploration')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load pre-trained model')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of episodes for final evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('rl_training')
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = make_env(args.env, seed=args.seed)
    eval_env = make_env(args.env, seed=args.seed + 1)
    
    logger.info(f"Environment: {args.env}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # Initialize agent
    if args.algorithm == 'q_learning':
        agent = QLearningAgent(
            env=env,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon
        )
    elif args.algorithm == 'dqn':
        agent = DQNAgent(
            env=env,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            device=device
        )
    elif args.algorithm == 'reinforce':
        agent = REINFORCEAgent(
            env=env,
            lr=args.lr,
            gamma=args.gamma,
            device=device
        )
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented yet")
    
    # Load pre-trained model if specified
    if args.load_model:
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    logger.info(f"Training {args.algorithm.upper()} for {args.episodes} episodes")
    
    # Train agent
    training_results = train_agent(
        agent=agent,
        env=env,
        episodes=args.episodes,
        render=args.render,
        logger=logger
    )
    
    # Evaluate agent
    logger.info(f"Evaluating agent for {args.eval_episodes} episodes")
    eval_results = evaluate_agent(
        agent=agent,
        env=eval_env,
        episodes=args.eval_episodes,
        render=False
    )
    
    logger.info(f"Average evaluation reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Save model if requested
    if args.save_model:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{args.algorithm}_{args.env}_{args.episodes}.pkl"
        agent.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    # Create visualizations
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    plot_training_results(
        training_results,
        save_path=results_dir / f"{args.algorithm}_{args.env}_training.png"
    )
    logger.info(f"Training plots saved to {results_dir}")
    
    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
