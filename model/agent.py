import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from models import PolicyNetwork, ValueNetwork

class PPOAgent:
    def __init__(self, input_shape, num_actions, 
                 gamma=0.99, 
                 epsilon=0.2, 
                 actor_lr=0.0003, 
                 critic_lr=0.001,
                 gae_lambda=0.95,
                 entropy_coef=0.01,
                 value_clip=0.2,
                 max_grad_norm=0.5,
                 ppo_epochs=4,
                 mini_batch_size=64):
        """
        Advanced PPO Agent with GAE, multiple epochs, entropy, etc.
        
        Args:
            input_shape: Dimension of state input
            num_actions: Number of possible actions
            gamma: Discount factor
            epsilon: PPO clipping parameter
            actor_lr: Learning rate for policy network
            critic_lr: Learning rate for value network
            gae_lambda: Lambda parameter for GAE
            entropy_coef: Entropy coefficient for exploration
            value_clip: Value function clipping parameter
            max_grad_norm: Maximum gradient norm for gradient clipping
            ppo_epochs: Number of optimization epochs per batch
            mini_batch_size: Size of mini-batches for optimization
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = PolicyNetwork(input_shape, num_actions).to(self.device)
        self.critic = ValueNetwork(input_shape).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Memory now stores entire trajectories rather than individual transitions
        self.trajectories = []
        self.current_trajectory = []
        
    def act(self, state):
        """Select an action from the policy distribution"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
            action_probs_np = action_probs.cpu().numpy()
        
        # Sample action based on probabilities
        action = np.random.choice(self.num_actions, p=action_probs_np[0])
        
        # Return action, probability, and value estimate
        value = self.critic(state).cpu().item()
        return action, action_probs[0][action].item(), value
    
    def remember(self, state, action, action_prob, reward, value, done):
        """Store transition in current trajectory"""
        self.current_trajectory.append((state, action, action_prob, reward, value, done))
        
        # If episode is done, store the trajectory and reset
        if done:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []
    
    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        return returns, advantages
    
    def update(self):
        """Perform PPO update using collected trajectories"""
        # If no complete trajectories, return
        if len(self.trajectories) == 0:
            return None, None, None
        
        # Flatten trajectories
        states, actions, old_probs, rewards, values, dones = [], [], [], [], [], []
        
        for trajectory in self.trajectories:
            # Get the last value for GAE calculation
            if not trajectory[-1][5]:  # If last step is not done
                # Get next state value for incomplete trajectory
                last_state = trajectory[-1][0]
                with torch.no_grad():
                    last_state_tensor = torch.FloatTensor(last_state).to(self.device)
                    last_value = self.critic(last_state_tensor).cpu().item()
            else:
                last_value = 0  # Terminal state has value 0
            
            # Extract data from trajectory
            traj_states = [t[0][0] for t in trajectory]  # Remove batch dimension
            traj_actions = [t[1] for t in trajectory]
            traj_old_probs = [t[2] for t in trajectory]
            traj_rewards = [t[3] for t in trajectory]
            traj_values = [t[4] for t in trajectory]
            traj_dones = [t[5] for t in trajectory]
            
            # Compute GAE and returns
            traj_returns, traj_advantages = self.compute_gae(
                traj_rewards, traj_values, last_value, traj_dones
            )
            
            # Extend lists
            states.extend(traj_states)
            actions.extend(traj_actions)
            old_probs.extend(traj_old_probs)
            rewards.extend(traj_rewards)
            values.extend(traj_values)
            dones.extend(traj_dones)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        returns = torch.FloatTensor(traj_returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(traj_advantages).unsqueeze(1).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset for mini-batch updates
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        # Track metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Perform multiple PPO epochs
        for _ in range(self.ppo_epochs):
            # Shuffle for mini-batch sampling
            np.random.shuffle(indices)
            
            # Process mini-batches
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:start_idx + self.mini_batch_size]
                
                # Get mini-batch data
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_probs = old_probs[batch_indices]
                mb_returns = returns[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_values = values[batch_indices]
                
                # Get current action probabilities and values
                mb_action_probs = self.actor(mb_states)
                mb_values_pred = self.critic(mb_states)
                
                # Get specific action probabilities
                batch_idx_tensor = torch.arange(len(batch_indices)).to(self.device)
                mb_new_probs = mb_action_probs[batch_idx_tensor, mb_actions]
                
                # Calculate ratio and surrogate objectives
                ratio = mb_new_probs / mb_old_probs
                
                # Policy loss with clipping
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss with clipping
                values_clipped = mb_values + torch.clamp(
                    mb_values_pred - mb_values, -self.value_clip, self.value_clip
                )
                value_loss_unclipped = (mb_values_pred - mb_returns).pow(2)
                value_loss_clipped = (values_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy for exploration
                entropy = -torch.sum(mb_action_probs * torch.log(mb_action_probs + 1e-10), dim=1).mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear trajectories after update
        self.trajectories = []
        
        # Compute average metrics
        num_updates = self.ppo_epochs * (dataset_size // self.mini_batch_size + 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def load(self, name):
        self.actor.load_state_dict(torch.load(f"{name}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{name}_critic.pth"))
    
    def save(self, name):
        torch.save(self.actor.state_dict(), f"{name}_actor.pth")
        torch.save(self.critic.state_dict(), f"{name}_critic.pth") 