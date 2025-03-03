import numpy as np
import time
from gymnasium.spaces import Space, Box
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen4EnvSinglePlayer, ObsType
import embedding.get_embeddings as get_embeddings

class SimpleRLPlayer(Gen4EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
    # Base reward from helper
        reward = self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, 
            victory_value=30.0, status_value=0.5  # Now rewarding status
        )
        
        # Reward for active Pokémon having good stats
        active_pokemon = current_battle.active_pokemon
        if active_pokemon and any(boost > 0 for boost in active_pokemon.boosts.values()):
            reward += 0.1  # Small bonus for having positive stat boosts
        
        # Reward entry hazards (spikes, stealth rock, etc.)
        if current_battle.side_conditions:
            reward += 0.2 * len(current_battle.side_conditions)
        
        # Compare with last battle state if available
        if last_battle:
            # Reward for applying status effects
            if (last_battle.opponent_active_pokemon and current_battle.opponent_active_pokemon and
                last_battle.opponent_active_pokemon.status is None and 
                current_battle.opponent_active_pokemon.status is not None):
                reward += 1.0  # Bonus for applying a status effect
            
            # Reward for setting up entry hazards
            if (len(current_battle.opponent_side_conditions) > 
                len(last_battle.opponent_side_conditions)):
                reward += 0.8  # Bonus for adding hazards
        
        return reward

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        return get_embeddings.GlorIA().embed_battle(battle)

    def describe_embedding(self) -> Space:
        """
        Provides an accurate description of the embedding space used by GlorIA embeddings.
        This ensures compatibility with Gymnasium's space validation.
        """
        # Create a 3022-dimensional space with appropriate bounds
        embedding_size = 3022
        
        # Initialize with zeros for minimum values
        low = np.zeros(embedding_size, dtype=np.float32)
        
        # Initialize with ones for binary features
        high = np.ones(embedding_size, dtype=np.float32)
        
        # Set specific bounds for ID-based features
        # For each Pokémon (starting at index 106)
        for i in range(12):
            base_idx = 106 + (i * 243)
            
            # Species ID bounds
            high[base_idx] = 296  # Number of Pokémon species in gen4
            
            # Ability ID bounds
            high[base_idx + 1] = 102  # Number of abilities in gen4
            
            # Item ID bounds
            high[base_idx + 2] = 38  # Number of items in gen4 + 1 for no item
            
            # Move ID bounds (4 moves)
            for j in range(4):
                high[base_idx + 3 + j] = 188  # Number of moves in gen4
            
            # Last used move bounds
            high[base_idx + 7] = 188  # Same as moves
        
        return Box(low, high, dtype=np.float32)
        
    def complete_current_battle(self):
        """
        Ensures the current battle is completed before moving on.
        This helps prevent idle battles from accumulating.
        """
        # Clear any pending battles
        self._battles = {}
        
        # Wait a moment to ensure everything is cleaned up
        time.sleep(1) 