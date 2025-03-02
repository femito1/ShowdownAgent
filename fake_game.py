from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.environment.effect import Effect
from poke_env.data.normalize import to_id_str
from src.gloria.embedding.get_embeddings import GlorIA 
import asyncio
import json

# The RandomPlayer is a basic agent that makes decisions randomly,
# serving as a starting point for more complex agent development.

print(to_id_str("Rock Head"))

ARR = []
color_change = False
PLAYER = GlorIA()
EMBED_JSON = {}

class DebugRandomPlayer(RandomPlayer):
    def choose_move(self, battle):
        # effects_here = []
        # if battle.active_pokemon.effects:
        #     effects_here = list(map(lambda x: x.value, battle.active_pokemon.effects.keys()))
        #     effects_there = list(map(lambda x: x.value, battle.opponent_active_pokemon.effects.keys()))
        # if 185 in effects_here or 185 in effects_there:
        #     print(f"FOUND TYPE CHANGE: {effects_here}")
        
        # check if u-turn or volt switch is used:
        # if battle.force_switch and not (battle.active_pokemon.fainted or battle.opponent_active_pokemon.fainted):
        #     pass
        
        EMBED_JSON[battle.turn] = PLAYER.test_embedding(battle)
        # PLAYER.test_embedding(battle)

        # if battle.turn == 6:
        #     with open("check_encoding.json", "w") as f:
        #         json.dump(EMBED_JSON, f)
        #     return ForfeitBattleOrder()
        # events = battle.observations[battle.turn-1].events
        # if (battle.active_pokemon.status and battle.active_pokemon.status.name == "TOX") or\
        #     (battle.opponent_active_pokemon.status and battle.opponent_active_pokemon.status.name == "TOX"):
        #     print(f"TOXIC: {battle.active_pokemon.status_counter} {battle.opponent_active_pokemon.status_counter}")

        # for event in events:
        #     if event[1] == "move" and event[3] in ("Encore", "Taunt"):
        #         pass
        ARR.append(battle)
        return self.choose_random_move(battle)
    
random_player = DebugRandomPlayer(battle_format="gen4randombattle")
second_random_player = RandomPlayer(battle_format="gen4randombattle")

# The battle_against method initiates a battle between two players.
# Here we are using asynchronous programming (await) to start the battle.
async def test():
    global color_change

    await random_player.battle_against(second_random_player, n_battles=1)
    
    print(f"battle {i}")
    if i%100 == 0:       
        with open("check_encoding_timeout.json", "a") as f:  # encoding data for a problematic battle
            json.dump(EMBED_JSON, f)
        
    
for i in range(1):
    asyncio.run(test())  # RUNNING THIS METHOD WILL SAVE THE EMBEDDING DATA FOR ONE BATTLE at a time

print("Fine.")
# with open("pokedex.json", "w") as f:
#     json.dump(GenData.from_gen(8).load_pokedex(gen=8), f)

# pokemons = GenData.from_gen(8).load_pokedex(gen=8)
# moves = GenData.from_gen(8).load_moves(gen=8)


# def get_abilities():
#     abilities = set()
#     for mon in pokemons:
#         for ability in pokemons[mon]["abilities"].values():
#             if ability:
#                 abilities.add(ability)
#     with open("abilities.json", "w") as f:
#         json.dump(dict(zip(sorted(abilities), range(len(abilities)))), f)



