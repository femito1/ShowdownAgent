import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from environment import SimpleRLPlayer
from agent import PPOAgent

def train_ppo_agent(num_episodes=6, batch_size=64, ppo_epochs=4, 
                   save_path=None, save_interval=50):
    """Train a PPO agent for Pokemon battles with advanced features"""
    
    # Configuration and environment setup with explicit player names
    agent_username = "GlorIAAgent"
    opponent_username = "RandomOpp"
    
    print(f"Setting up players: {agent_username} vs {opponent_username}")
    
    # Create account configurations with consistent names
    agent_config = AccountConfiguration(agent_username, None)
    opponent_config = AccountConfiguration(opponent_username, None)
    
    # Create players with explicit names
    opponent = RandomPlayer(
        battle_format="gen4randombattle",
        account_configuration=opponent_config
    )
    
    train_env = SimpleRLPlayer(
        battle_format="gen4randombattle", 
        opponent=opponent, 
        start_challenging=False,
        account_configuration=agent_config
    )

    print("Initializing environment and agent...")
    input_shape = 2998
    num_actions = train_env.action_space_size()
    print(f"Action space size: {num_actions}")
    
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
    
    episode_rewards = []
    episode_lengths = []
    train_env.reset()
    # Start with just one challenge at a time and ensure we wait for it to complete
    print(f"Starting training with {agent_username} vs {opponent_username}")
    
    for e in range(1, num_episodes + 1):
        print(f"\nStarting episode {e}/{num_episodes}")
        print("Starting a new challenge...")
        train_env.reset()
        # Start a single challenge for this episode
        train_env._stop_challenge_loop()  # Stop any existing challenges
        train_env.start_challenging(n_challenges=1)
        

        print("Battle is ready, getting initial state...")
        initial_state = train_env.embed_battle(train_env.current_battle)
        print(f"Initial state shape: {initial_state.shape}")
        
        state = np.reshape(initial_state, [1, agent.input_shape])
        done = False
        steps = 0
        episode_reward = 0
        
        # Run the episode
        print("Starting episode loop...")
        while not done:
            # Get action, probability, and value estimate
            action, action_prob, value = agent.act(state)
            print(f"Step {steps+1}: Taking action {action}")
            
            # Take action in the environment
            next_state, reward, done, _, _ = train_env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            
            next_state = np.reshape(next_state, [1, agent.input_shape])
            
            # Store transition
            agent.remember(state, action, action_prob, reward, value, done)
            
            state = next_state
            steps += 1
            episode_reward += reward
        
        print("Waiting before next episode...")
        
        # Track episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Ensure we stop challenging before the next episode
        print("Stopping challenge loop...")
        train_env._stop_challenge_loop()
        train_env.complete_current_battle()
        
        # Update policy after each episode
        print("Updating policy...")
        policy_loss, value_loss, entropy = agent.update()
        
        # Print progress
        print(f"Episode {e}/{num_episodes}, Steps: {steps}, Reward: {episode_reward:.2f}")
        if policy_loss is not None:
            print(f"  Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Periodically save the model
        if save_path and e % save_interval == 0:
            print(f"Saving model checkpoint at episode {e}...")
            agent.save(f"{save_path}_episode_{e}")
            

        
    

    # Reset environment
    print("Training complete, cleaning up...")
    train_env.reset_env(restart=False)
    
    # Explicitly stop challenging and wait for battles to complete
    print("Stopping challenge loop...")
    train_env._stop_challenge_loop()
    train_env.complete_current_battle()
    train_env.close()
    
    # Save the final trained model
    if save_path:
        print(f"Saving final model to {save_path}...")
        agent.save(save_path)
    
    # Print training summary
    mean_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths[-20:]) if len(episode_lengths) >= 20 else np.mean(episode_lengths)
    
    print("\n===== TRAINING SUMMARY =====")
    print(f"Episodes completed: {len(episode_rewards)}/{num_episodes}")
    print(f"Average reward (last 20 episodes): {mean_reward:.2f}")
    print(f"Average episode length (last 20 episodes): {mean_length:.2f}")
    
    return agent

