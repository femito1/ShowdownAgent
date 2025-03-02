import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen4EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer
)
import src.gloria.embedding.get_embeddings as get_embeddings
from poke_env import AccountConfiguration
from gymnasium.spaces import Space, Box

class SimpleRLPlayer(Gen4EnvSinglePlayer):
    def calc_reward(self, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        return get_embeddings.GlorIA().embed_battle(battle)

    def describe_embedding(self) -> Space: #still needs to be updated
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

# Create PyTorch policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.model(x)

# Create PyTorch value network
class ValueNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class PPOAgent:
    def __init__(self, input_shape, num_actions, gamma=0.95, epsilon=0.2, actor_lr=0.0003, critic_lr=0.001):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon  # Clipping value for PPO
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = PolicyNetwork(input_shape, num_actions).to(self.device)
        self.critic = ValueNetwork(input_shape).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.memory = deque(maxlen=2000)
    
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
            action_probs_np = action_probs.cpu().numpy()
        
        # Sample action based on probabilities
        action = np.random.choice(self.num_actions, p=action_probs_np[0])
        
        # Return both the action and its probability
        return action, action_probs[0][action].item()
    
    def remember(self, state, action, action_prob, reward, next_state, done):
        # Store the action probability at the time the action was taken
        self.memory.append((state, action, action_prob, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        # Collect batch data
        states = []
        actions = []
        old_probs = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in minibatch:
            state, action, old_prob, reward, next_state, done = experience
            states.append(state[0])  # Remove the extra dimension
            actions.append(action)
            old_probs.append(old_prob)
            rewards.append(reward)
            next_states.append(next_state[0])  # Remove the extra dimension
            dones.append(done)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic update (process entire batch at once)
        self.critic_optimizer.zero_grad()
        
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        # Calculate targets and advantages for all samples at once
        targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = targets - values
        
        # Compute critic loss on the entire batch
        critic_loss = nn.MSELoss()(values, targets.detach())
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update (process entire batch at once)
        self.actor_optimizer.zero_grad()
        
        # Get action probabilities for all actions in the batch
        action_probs = self.actor(states)
        batch_indices = torch.arange(actions.size(0)).to(self.device)
        selected_probs = action_probs[batch_indices, actions]
        
        # Calculate ratios for all samples at once
        ratios = selected_probs / old_probs
        
        # Calculate surrogate losses
        surrogate1 = ratios * advantages.detach()
        surrogate2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages.detach()
        
        # Calculate actor loss (negative because we're maximizing)
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()
    
    def load(self, name):
        self.actor.load_state_dict(torch.load(f"{name}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{name}_critic.pth"))
    
    def save(self, name):
        torch.save(self.actor.state_dict(), f"{name}_actor.pth")
        torch.save(self.critic.state_dict(), f"{name}_critic.pth")

# Configuration and environment setup
config2, config5 = AccountConfiguration("Opp2", None), AccountConfiguration("GG2", None)
opponent = RandomPlayer(battle_format="gen4randombattle",
                        account_configuration=config2)
train_env = SimpleRLPlayer(
    battle_format="gen4randombattle", opponent=opponent, start_challenging=False,
    account_configuration=config5
)

# Compute dimensions
input_shape = 3023 # subject to change

# Training loop
num_episodes = 6
batch_size = 2
num_actions = train_env.action_space_size()
agent = PPOAgent(input_shape=input_shape, num_actions=num_actions)
train_env.start_challenging(n_challenges=num_episodes)

for e in range(1, num_episodes + 1):
    train_env.reset()
    initial_state = train_env.embed_battle(train_env.current_battle)
    state = np.reshape(initial_state, [1, agent.input_shape])
    done = False
    time = 0
    episode_reward = 0
    
    while not done:
        action, action_prob = agent.act(state)
        
        next_state, reward, done, _, info = train_env.step(action)
        next_state = np.reshape(next_state, [1, agent.input_shape])
        
        agent.remember(state, action, action_prob, reward, next_state, done)
        
        state = next_state
        time += 1
        episode_reward += reward
        
        if done:
            print(f"Episode {e}/{num_episodes}, steps: {time}, reward: {episode_reward:.2f}")
    
    # Perform batch optimization
    if len(agent.memory) > batch_size:
        critic_loss, actor_loss = agent.replay(batch_size)
        print(f"  Losses - Critic: {critic_loss:.4f}, Actor: {actor_loss:.4f}")

# Reset environment
train_env.reset_env(restart=False)
train_env.close()

# Test Function
def test(agent, environment, nb_episodes=100):
    victories = 0
    for e in range(nb_episodes):
        environment.reset()
        s = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
        done = False
        while not done:
            action, _ = agent.act(s)  # Only use the action, ignore probability during testing
            next_state, reward, done, _, info = environment.step(action)
            next_state = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
            s = next_state
            if done:
                if reward > 0:  # Assuming a positive reward indicates a win
                    victories += 1
                print(f"Episode {e + 1}/{nb_episodes} finished. Reward: {reward}")
    print(f"Test completed: {victories}/{nb_episodes} victories")
    environment.close()

# Players and Environments setup (ignore) --------------------------------------------------------------
opponent = RandomPlayer(battle_format="gen4randombattle",
                        account_configuration=config3)
eval_env = SimpleRLPlayer(
    battle_format="gen4randombattle", opponent=opponent, start_challenging=True,
    account_configuration=config6
)
opp_maxi = AccountConfiguration("max", None)
opp_heur = AccountConfiguration("heur", None)
maxi = MaxBasePowerPlayer(battle_format="gen4randombattle",
                          account_configuration=opp_maxi)
heur = SimpleHeuristicsPlayer(battle_format="gen4randombattle",
                              account_configuration=opp_heur)
eval2 = AccountConfiguration("trained_vs_maxi", None)
eval3 = AccountConfiguration("trained_vs_heur", None)
eval_env2 = SimpleRLPlayer(battle_format="gen4randombattle", opponent=maxi, start_challenging=True,
                           account_configuration=eval2)
eval_env3 = SimpleRLPlayer(battle_format="gen4randombattle", opponent=heur, start_challenging=True,
                           account_configuration=eval3)
#--------------------------------------------------------------------------------------------------------

# Saving would be done with:
# agent.save("ppo_model")
