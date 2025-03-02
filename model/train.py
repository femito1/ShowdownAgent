import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from environment import SimpleRLPlayer
from agent import PPOAgent

def train_ppo_agent(num_episodes=6, batch_size=64, ppo_epochs=4, 
                   save_path=None, save_interval=10):
    """Train a PPO agent for Pokemon battles with advanced features"""
    
    # Configuration and environment setup
    config2, config5 = AccountConfiguration("Opp2", None), AccountConfiguration("GG2", None)
    opponent = RandomPlayer(battle_format="gen4randombattle",
                            account_configuration=config2)
    train_env = SimpleRLPlayer(
        battle_format="gen4randombattle", opponent=opponent, start_challenging=False,
        account_configuration=config5
    )

    # Compute dimensions
    input_shape = 3023  # subject to change based on your embedding size
    num_actions = train_env.action_space_size()
    
    # Create agent with advanced PPO features
    agent = PPOAgent(
        input_shape=input_shape, 
        num_actions=num_actions,
        gamma=0.99,
        epsilon=0.2,
        actor_lr=0.0003,
        critic_lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_clip=0.2,
        max_grad_norm=0.5,
        ppo_epochs=ppo_epochs,
        mini_batch_size=batch_size
    )
    
    # Start training
    train_env.start_challenging(n_challenges=num_episodes)
    
    episode_rewards = []
    episode_lengths = []

    for e in range(1, num_episodes + 1):
        train_env.reset()
        initial_state = train_env.embed_battle(train_env.current_battle)
        state = np.reshape(initial_state, [1, agent.input_shape])
        done = False
        steps = 0
        episode_reward = 0
        
        # Run the episode
        while not done:
            # Get action, probability, and value estimate
            action, action_prob, value = agent.act(state)
            
            # Take action in the environment
            next_state, reward, done, _, info = train_env.step(action)
            next_state = np.reshape(next_state, [1, agent.input_shape])
            
            # Store transition
            agent.remember(state, action, action_prob, reward, value, done)
            
            state = next_state
            steps += 1
            episode_reward += reward
            
        # Track episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Update policy after each episode
        policy_loss, value_loss, entropy = agent.update()
        
        # Print progress
        print(f"Episode {e}/{num_episodes}, Steps: {steps}, Reward: {episode_reward:.2f}")
        if policy_loss is not None:
            print(f"  Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Periodically save the model
        if save_path and e % save_interval == 0:
            agent.save(f"{save_path}_episode_{e}")

    # Reset environment
    train_env.reset_env(restart=False)
    train_env.close()
    
    # Save the final trained model
    if save_path:
        agent.save(save_path)
    
    # Print training summary
    mean_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths[-20:]) if len(episode_lengths) >= 20 else np.mean(episode_lengths)
    
    print("\n===== TRAINING SUMMARY =====")
    print(f"Episodes completed: {num_episodes}")
    print(f"Average reward (last 20 episodes): {mean_reward:.2f}")
    print(f"Average episode length (last 20 episodes): {mean_length:.2f}")
    
    return agent

if __name__ == "__main__":
    # Train with advanced PPO settings
    agent = train_ppo_agent(
        num_episodes=200,  # More episodes for better learning
        batch_size=128,    # Larger batch size for more stable updates
        ppo_epochs=10,     # Multiple optimization epochs per episode
        save_path="advanced_ppo_model",
        save_interval=20
    ) 