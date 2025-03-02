import json
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import Battle, Pokemon, Weather
from poke_env.player import Gen4EnvSinglePlayer
import numpy as np
from gymnasium.spaces import Box

DATA_DIR = "src/gloria/embedding/data/"  # please nand run the code from the root directory of the project

with open(DATA_DIR + "gen4randombattle.json", "r") as f:
    GEN4 = json.load(f)
    # print(f"pokemon gen4: {len(GEN4)}")


def get_pokemons(gen):
    pokemons = dict(
        zip(
            sorted(map(lambda x: to_id_str(x), gen.keys())),
            [i + 1 for i in range(len(gen))],
        )
    )
    with open(DATA_DIR + "gen4pokemon.json", "w") as f:
        json.dump(pokemons, f)
    return pokemons


def count_abilities(gen):
    abilities = set()
    for mon in gen:
        for ability in gen[mon]["abilities"]:
            abilities.add(to_id_str(ability))
    abilities.add("oblivious")
    sorted_abilities = dict(
        zip(sorted(abilities), [i + 1 for i in range(len(abilities))])
    )
    with open(DATA_DIR + "gen4abilities.json", "w") as f:
        json.dump(sorted_abilities, f)
    return sorted_abilities


def count_items(gen):
    items = set()
    for mon in gen:
        for item in gen[mon]["items"]:
            items.add(to_id_str(item))
    sorted_items = dict(zip(sorted(items), [i + 1 for i in range(len(items))]))
    with open(DATA_DIR + "gen4items.json", "w") as f:
        json.dump(sorted_items, f)
    return sorted_items


def count_moves(gen):
    moves = set()
    for mon in gen:
        for role in gen[mon]["roles"]:
            for move in gen[mon]["roles"][role]["moves"]:
                moves.add(to_id_str(move))
    moves.add("struggle")
    moves.add("hiddenpower")
    sorted_moves = dict(zip(sorted(moves), [i + 1 for i in range(len(moves))]))

    with open(DATA_DIR + "gen4moves.json", "w") as f:
        json.dump(sorted_moves, f)
    return sorted_moves


# count_abilities(GEN4)
# count_items(GEN4)
# count_moves(GEN4)
# get_pokemons(GEN4)
# when we get types during battle, subtract 1 if the value is greater than 4 since fairy is not in gen4


def get_unknown_pokemon():
    species = np.array([0])  # EMBEDDING
    ability = np.array([0])  # EMBEDDING
    item = np.array([0])  # EMBEDDING

    pp = np.zeros(16)
    moves_encoding = np.array([0, 0, 0, 0])  # EMBEDDING
    last_used_move = np.array([0])  # EMBEDDING

    type1 = np.zeros(17)
    type2 = np.zeros(18)  # can be null type

    hp = np.zeros(17)
    hp[0] = 1  # unknown health is full health

    boosts_encoding = np.zeros(84)

    effects_encoding = np.zeros(19)
    taunt = np.zeros(5)
    encore = np.zeros(8)
    slow_start = np.zeros(5)

    # gender = np.zeros(3)
    trapped = np.array([0])
    status = np.zeros(7)

    toxic_counter = np.zeros(15)
    sleep_counter = np.zeros(4)

    weight_encoding = np.zeros(6)

    first_turn = np.zeros(1)

    protect_counter = np.zeros(5)

    is_mine = np.array([0])
    must_recharge = np.array([0])
    preparing = np.array([0])
    active = np.array([0])
    unknown = np.array([1])
    return_vector = np.concatenate(
        [
            species,
            ability,
            item,
            moves_encoding,
            last_used_move,
            pp,
            type1,
            type2,
            hp,
            boosts_encoding,
            effects_encoding,
            taunt,
            encore,
            slow_start,
            # gender,
            trapped,
            status,
            toxic_counter,
            sleep_counter,
            weight_encoding,
            first_turn,
            protect_counter,
            is_mine,
            must_recharge,
            preparing,
            active,
            unknown,
        ],
        dtype=np.float32,
    )
    return return_vector


# def get_bounds():
#     pass


# DESCRIBED_EMBEDDING = get_bounds()
UNKNOWN_POKEMON = get_unknown_pokemon()

# Load dicts for encoding
with open(DATA_DIR + "gen4abilities.json", "r") as f:
    ABILITIES = json.load(f)
with open(DATA_DIR + "gen4items.json", "r") as f:
    ITEMS = json.load(f)
with open(DATA_DIR + "gen4moves.json", "r") as f:
    MOVES = json.load(f)
with open(DATA_DIR + "gen4pokemon.json", "r") as f:
    POKEMONS = json.load(f)
with open(DATA_DIR + "gen4effects.json", "r") as f:
    EFFECTS = json.load(f)


# weather = {"HAIL": np.zeros(9),
#            "RAINDANCE": np.zeros(9),
#            "SANDSTORM": np.zeros(9)}


class GlorIA(Gen4EnvSinglePlayer):  # not inhereting from Gen4EnvSinglePlayer temorarily to test the embed_battle method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_move_dict = {}

    def test_embedding(self, battle: Battle):
        arr = self.embed_battle(battle).tolist()
        return arr

    def get_pkmn_battle_id(self, pkmn_id):
        return f"{pkmn_id[:2]}a{pkmn_id[2:]}"


    def calc_reward(self, current_battle) -> float:
        return self.reward_computing_helper(current_battle, victory_value=1.0)

    def describe_embedding(self):
        EMBEDDED_VECTOR_MIN = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ] 
        EMBEDDED_VECTOR_MAX = [
            296,
            102,
            38,
            188,
            188,
            188,
            188,
            188,
        ] 
        low = np.concatenate(
            [np.zeros(106), [*EMBEDDED_VECTOR_MIN, *np.zeros(233)] * 12]
        )

        high = np.concatenate(
            [np.ones(106), [*EMBEDDED_VECTOR_MAX, *np.ones(233)] * 12]
        )
        return Box(low, high, dtype=np.float32)

    def embed_battle(self, battle: Battle):
        """
        hail, rain, sandstorm, sun, t_room, force_switch, n_unknown, stealth_rock, spikes, toxic_spikes
        """
        turn = battle.turn
        if turn == 1:  # initializing last used move dict
            self.last_move_dict = {}
        self.update_last_used_move(
            battle.observations[battle.turn - 1].events
        )  # TODO: CONTROLLARE SE è TURN-1 O TURN
        hail, rain, sandstorm, sun = self.get_weather(battle)
        t_room = np.zeros(5)
        if battle.fields:
            start_turn = battle.fields.get("trickroom", 0)
            if start_turn:
                index = min(start_turn - turn, -1)
                t_room[index] = 1
        force_switch = np.array([int(battle.force_switch)])
        n_unknown = np.zeros(6)
        n_unknown[-len(battle.opponent_team)] = 1
        lightscreen, reflect, safeguard, spikes, stealth_rocks, toxic_spikes = (
            self.get_side_conditions(battle)
        )
        pokemons_to_encode = self.encode_pokemons(battle)
        return_vector = np.concatenate(
            [
                sun,
                rain,
                hail,
                sandstorm,
                t_room,
                force_switch,
                n_unknown,
                stealth_rocks,
                spikes,
                toxic_spikes,
                reflect,
                lightscreen,
                safeguard,
                pokemons_to_encode,
            ]
        )
        return return_vector

    def get_weather(self, battle: Battle):
        turn = battle.turn
        sun = rain = hail = sandstorm = np.zeros(9)
        if battle.weather:
            weather, starting_turn = next(iter(battle.weather.items()))
            weather = weather.name
            index = min(-min(turn - starting_turn, 9), -1)
            me, opponent = battle.all_active_pokemons
            active_abilities = (me.ability, opponent.ability)
            if weather == "SUNNYDAY":
                if "drought" in active_abilities:
                    sun[0] = 1
                else:
                    sun[index] = 1
            elif weather == "RAINDANCE":
                if "drizzle" in active_abilities:
                    rain[0] = 1
                else:
                    rain[index] = 1
            elif weather == "HAIL":
                if "snowwarning" in active_abilities:
                    hail[0] = 1
                else:
                    hail[index] = 1
            elif weather == "SANDSTORM":
                if "sandstream" in active_abilities:
                    sandstorm[0] = 1
                else:
                    sandstorm[index] = 1
        return hail, rain, sandstorm, sun

    def get_side_conditions(self, battle: Battle):
        turn = battle.turn
        me = battle.side_conditions
        opponent = battle.opponent_side_conditions
        me_lightscreen = me.get("lightscreen", False)
        opponent_lightscreen = opponent.get("lightscreen", False)
        lightscreen = np.zeros(18)
        if me_lightscreen:
            me_index = min(me_lightscreen - turn, -1)
            lightscreen[me_index] = 1
        if opponent_lightscreen:
            opponent_index = min(opponent_lightscreen - turn, -1) - 9
            lightscreen[opponent_index] = 1
        me_reflect = me.get("reflect", False)
        opponent_reflect = opponent.get("reflect", False)
        reflect = np.zeros(18)
        if me_reflect:
            me_index = min(me_reflect - turn, -1)
            reflect[me_index] = 1
        if opponent_reflect:
            opponent_index = min(opponent_reflect - turn, -1) - 9
            reflect[opponent_index] = 1
        me_safeguard = me.get("safeguard", False)
        opponent_safeguard = opponent.get("safeguard", False)
        safeguard = np.zeros(10)
        if me_safeguard:
            me_index = min(me_safeguard - turn, -1)
            safeguard[me_index] = 1
        if opponent_safeguard:
            opponent_index = min(opponent_safeguard - turn, -1) - 5
            safeguard[opponent_index] = 1
        me_spikes = me.get("spikes", 0)
        opponent_spikes = opponent.get("spikes", 0)
        spikes = np.zeros(6)
        if me_spikes:
            spikes[-me_spikes] = 1
        if opponent_spikes:
            spikes[-opponent_spikes - 3] = 1
        me_stealth_rocks = me.get("stealthrock", False)
        opponent_stealth_rocks = opponent.get("stealthrock", False)
        stealth_rocks = np.zeros(2)
        if me_stealth_rocks:
            stealth_rocks[1] = 1
        if opponent_stealth_rocks:
            stealth_rocks[0] = 1
        me_toxic_spikes = me.get("toxicspikes", 0)
        opponent_toxic_spikes = opponent.get("toxicspikes", 0)
        toxic_spikes = np.zeros(4)
        if me_toxic_spikes:
            toxic_spikes[-me_toxic_spikes] = 1
        if opponent_toxic_spikes:
            toxic_spikes[-opponent_toxic_spikes - 2] = 1
        return lightscreen, reflect, safeguard, spikes, stealth_rocks, toxic_spikes

    def get_pp_bin(self, pp: int):
        if not pp:
            return 0
        elif 1 <= pp <= 7:
            return -1
        elif 8 <= pp <= 26:
            return -2
        else:
            return -3

    def update_last_used_move(self, events: list):
        for event in events:
            if event[1] == "move":
                pkmn_id = event[2]
                move_id = MOVES[to_id_str(event[3])]
                if len(self.last_move_dict) < 12:
                    self.last_move_dict[pkmn_id] = move_id
                else:
                    self.last_move_dict = {pkmn_id: move_id}
                assert (
                    len(self.last_move_dict) <= 12
                ), f"Il dizionario delle last used move ha ecceduto il \
                    numero di pokémon totali \n\n{self.last_move_dict}"

    def get_weight_bin(self, weight: float):
        if weight < 10:
            return -1
        elif 10 <= weight < 25:
            return -2
        elif 25 <= weight < 50:
            return -3
        elif 50 <= weight < 100:
            return -4
        elif 100 <= weight < 200:
            return -5
        else:
            return -6

    def encode_pokemons(self, battle: Battle):
        pokemons_encoding = []
        all_mons = battle.team | battle.opponent_team
        for mon_name in all_mons:
            mon: Pokemon = all_mons[mon_name]
            if not mon.active:
                self.last_move_dict[self.get_pkmn_battle_id(mon_name)] = 0
            opponent = battle.opponent_role == mon_name[:2]

            # handling for letters of uknown
            species = mon.species
            for s in ("unown", "gastrodon"):
                if s in species:
                    species = s
                    break

            species = np.array([POKEMONS[species]])  # EMBEDDING
            ability = np.array(
                [ABILITIES[mon.ability] if mon.ability else 0]
            )  # EMBEDDING
            if mon.item:
                item_number = ITEMS[mon.item] if mon.item != "unknown_item" else 0
            else:
                item_number = len(ITEMS) + 1
            item = np.array([item_number])  # EMBEDDING
            moves = mon.moves
            pp = np.zeros(16)
            moves_encoding = np.array(
                list(map(lambda x: MOVES[x], moves.keys())) + [0] * (4 - len(moves))
            )  # EMBEDDING
            for i, move in enumerate(moves):
                pp_bin = self.get_pp_bin(moves[move].current_pp)
                pp[pp_bin - (3 * i)]
            last_used_move = np.array(
                [self.last_move_dict.get(self.get_pkmn_battle_id(mon_name), 0)]
            )  # EMBEDDING

            type1 = np.zeros(17)
            type2 = np.zeros(18)  # can be null type
            # SOPRA AL 5 TOGLI 1, TOTALE 17 TIPI manca fairy (gen6)
            get_type_index = lambda x: x - 1 if x > 4 else x
            type1[-get_type_index(mon.type_1.value)]
            if mon.type_2:
                type2[-get_type_index(mon.type_2.value)]
            else:
                type2[0] = 1

            hp = np.zeros(17)
            if mon.current_hp_fraction:
                hp_bin = -int(
                    mon.current_hp_fraction * 100 / 6.25
                )  # to have 17 bins (16th of health + max)
                hp[hp_bin] = 1

            boosts_encoding = np.zeros(84)
            for i, stat in enumerate(mon.boosts):
                value = mon.boosts[stat]
                if value > 0:
                    boosts_encoding[value + 5 + (12 * i)]
                elif value < 0:
                    boosts_encoding[value + 6 + (12 * i)]

            effects_encoding = np.zeros(19)
            taunt = np.zeros(5)
            encore = np.zeros(8)
            slow_start = np.zeros(5)

            for effect in mon.effects:
                if effect.name in EFFECTS:
                    effects_encoding[EFFECTS[effect.name]] = 1
                elif effect.name == "TAUNT":
                    turn = mon.effects[effect]
                    taunt[-turn] = 1
                elif effect.name == "ENCORE":
                    turn = mon.effects[effect]
                    encore[-turn] = 1
                elif effect.name == "SLOW_START":
                    turn = mon.effects[effect]
                    slow_start[-turn] = 1

            # gender = np.zeros(3)
            # gender[-mon.gender.value] = 1
            trapped = np.array([int(battle.trapped)])
            status = np.zeros(7)
            if mon.status:
                status[-mon.status.value] = 1

            toxic_counter = np.zeros(15)
            sleep_counter = np.zeros(4)
            if mon.status_counter:
                if mon.status:
                    if mon.status.name == "SLP":
                        sleep_counter[max(-mon.status_counter, -4)] = 1
                    elif mon.status.name == "TOX":
                        toxic_counter[max(-mon.status_counter, -15)] = 1

            weight_encoding = np.zeros(6)
            weight_encoding[self.get_weight_bin(mon.weight)] = 1

            first_turn = np.array([int(mon.first_turn)])

            protect_counter = np.zeros(5)
            if mon.protect_counter:
                protect_counter[-mon.protect_counter] = 1

            is_mine = np.array([int(not opponent)])
            must_recharge = np.array([int(mon.must_recharge)])
            preparing = np.array([int(mon.preparing)])
            active = np.array([int(mon.active)])
            unknown = np.array([0])
            pokemons_encoding.append(
                np.concatenate(
                    [
                        species,
                        ability,
                        item,
                        moves_encoding,
                        last_used_move,
                        pp,
                        type1,
                        type2,
                        hp,
                        boosts_encoding,
                        effects_encoding,
                        taunt,
                        encore,
                        slow_start,
                        # gender,
                        trapped,
                        status,
                        toxic_counter,
                        sleep_counter,
                        weight_encoding,
                        first_turn,
                        protect_counter,
                        is_mine,
                        must_recharge,
                        preparing,
                        active,
                        unknown,
                    ],
                    dtype=np.float32,
                )
            )
        # Standard pokemon encoding for unknown pokemons
        for i in range(12 - len(pokemons_encoding)):
            pokemons_encoding.append(
                UNKNOWN_POKEMON
            )  # all unknown pokemons are the same

        return np.concatenate(pokemons_encoding, dtype=np.float32)