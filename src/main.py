from train import train_ppo_agent
from evaluate import evaluate_against_opponents
import time

def main():
    # Training phase
    print("======= TRAINING PHASE =======")
    agent = train_ppo_agent(
        num_episodes=6,  # Increase for better performance
        batch_size=64,     # Larger batch size for more stable learning
        save_path="ppo_model"
    )
    
    # Add a delay to ensure all battles are properly completed
    print("\nWaiting for all battles to complete before starting evaluation...")
    time.sleep(5)  # 5 second delay
    
    # Evaluation phase
    print("\n======= EVALUATION PHASE =======")
    #results = evaluate_against_opponents(agent)
    
    # You could save the results to a file, plot them, etc.
    
    print("\nTraining and evaluation complete!")

main()