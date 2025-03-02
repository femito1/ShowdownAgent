from train import train_ppo_agent
from evaluate import evaluate_against_opponents
from agent import PPOAgent

def main():
    # Training phase
    print("======= TRAINING PHASE =======")
    agent = train_ppo_agent(
        num_episodes=100,  # Increase for better performance
        batch_size=64,     # Larger batch size for more stable learning
        save_path="ppo_model"
    )
    
    # Evaluation phase
    print("\n======= EVALUATION PHASE =======")
    results = evaluate_against_opponents(agent)
    
    # You could save the results to a file, plot them, etc.
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main() 