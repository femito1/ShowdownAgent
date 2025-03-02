import numpy as np
from poke_env.environment import Battle, Move, Pokemon
from poke_env.player import BattleOrder
from poke_env.data.gen_data import GenData
from gloria.embedding.get_embeddings import MOVES, POKEMONS

class Simulator():
    def __init__(self):
        data = GenData(4)
        '''
        self.moves = dict(m for m in data.moves.items() if m[0] in MOVES.keys())
        self.pokemon = dict(m for m in data.pokedex.items() if m[0] in POKEMONS.keys())
        '''

        self.rd_moves = list(MOVES.keys()) # we use the index as the key from now on
        self.rd_pokemon = list(POKEMONS.keys())

    def sim(self, battle: Battle, action: int) -> Battle:
        def decode_action(action_idx: int) -> BattleOrder:
            '''
            decode mask of the available action to a BattleOrder type object
            '''
            if action_idx >= len(self.rd_moves): # it is a switch
                action_idx -= len(self.rd_moves) # IF WE STORE THE MASK AS MOVES CONCATENATE POKEMON  
                switch = self.rd_pokemon[action_idx]
                action: Pokemon = [p for p in battle.available_switches if p.base_species == switch][0] 
            else:
                move = self.rd_moves[action_idx]
                action: Move = [m for m in battle.available_moves if m.id() == move][0] 
            
            order = BattleOrder(action)
            return order
        
        action = decode_action(action)