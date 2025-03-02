import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer
)
from environment import SimpleRLPlayer

def test(agent, environment, nb_episodes=100):
    """Test a trained agent against a specific environment/opponent"""
    victories = 0
    total_reward = 0
    
    for e in range(nb_episodes):
        environment.reset()
        s = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.act(s)  # Only use the action, ignore probability during testing
            next_state, reward, done, _, info = environment.step(action)
            next_state = np.reshape(environment.embed_battle(environment.current_battle), [1, agent.input_shape])
            s = next_state
            episode_reward += reward
            
            if done:
                if reward > 0:  # Assuming a positive reward indicates a win
                    victories += 1
                total_reward += episode_reward
                print(f"Episode {e + 1}/{nb_episodes} finished. Reward: {episode_reward:.2f}")
    
    print(f"Test results: {victories}/{nb_episodes} victories ({victories/nb_episodes*100:.1f}%)")
    print(f"Average reward: {total_reward/nb_episodes:.2f}")
    
    environment.close()
    return victories/nb_episodes, total_reward/nb_episodes

def evaluate_against_opponents(agent, battle_format="gen4randombattle"):
    """Evaluate an agent against multiple standard opponents"""
    
    # Setup account configurations
    random_config = AccountConfiguration("random_opp", None)
    max_power_config = AccountConfiguration("max_power", None)
    heuristic_config = AccountConfiguration("heuristic", None)
    agent_config = AccountConfiguration("agent", None)
    
    # Create opponents
    random_player = RandomPlayer(battle_format=battle_format, account_configuration=random_config)
    max_power_player = MaxBasePowerPlayer(battle_format=battle_format, account_configuration=max_power_config)
    heuristic_player = SimpleHeuristicsPlayer(battle_format=battle_format, account_configuration=heuristic_config)
    
    # Create evaluation environments
    random_env = SimpleRLPlayer(
        battle_format=battle_format, opponent=random_player, 
        start_challenging=True, account_configuration=agent_config
    )
    max_power_env = SimpleRLPlayer(
        battle_format=battle_format, opponent=max_power_player, 
        start_challenging=True, account_configuration=agent_config
    )
    heuristic_env = SimpleRLPlayer(
        battle_format=battle_format, opponent=heuristic_player, 
        start_challenging=True, account_configuration=agent_config
    )
    
    # Run evaluations
    print("Evaluating against Random player...")
    random_win_rate, random_reward = test(agent, random_env, nb_episodes=50)
    
    print("\nEvaluating against Max Power player...")
    maxpower_win_rate, maxpower_reward = test(agent, max_power_env, nb_episodes=50)
    
    print("\nEvaluating against Heuristic player...")
    heuristic_win_rate, heuristic_reward = test(agent, heuristic_env, nb_episodes=50)
    
    # Display summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"vs Random: {random_win_rate*100:.1f}% win rate, {random_reward:.2f} avg reward")
    print(f"vs Max Power: {maxpower_win_rate*100:.1f}% win rate, {maxpower_reward:.2f} avg reward")
    print(f"vs Heuristic: {heuristic_win_rate*100:.1f}% win rate, {heuristic_reward:.2f} avg reward")
    
    return {
        "random": (random_win_rate, random_reward),
        "max_power": (maxpower_win_rate, maxpower_reward),
        "heuristic": (heuristic_win_rate, heuristic_reward)
    } 