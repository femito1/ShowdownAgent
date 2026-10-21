"""Battle -> flat float32 vector.

Layout (token-major, consumed by `porygon.model.BattleTransformer`):
    [field token | 6 own mons (battle.team order = switch action slots) | 6 opponent mons]
Each mon token is [categorical ids | movepool ids | mon numerics | 4 x move numerics].
Categorical ids are stored as floats and embedded by the model; 0 = none/padding, 1 = unknown.

Following Wang (MIT 2024), every multi-turn condition carries its elapsed-turn count so the
state is close to Markov without a recurrent policy.
"""

import json
import math
from functools import cache
from pathlib import Path

import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData

DATA = Path(__file__).parent / "data"
VOCAB = json.loads((DATA / "vocab.json").read_text())
PAD, UNK = 0, 1
INDEX = {k: {e: i + 2 for i, e in enumerate(v)} for k, v in VOCAB.items()}
VOCAB_SIZES = {k: len(v) + 2 for k, v in VOCAB.items()}

TYPES = [t for t in PokemonType if t not in (PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR)]
STATUSES = [s for s in Status if s is not Status.FNT]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
EFFECTS = [
    Effect.CONFUSION, Effect.SUBSTITUTE, Effect.LEECH_SEED, Effect.TAUNT, Effect.ENCORE,
    Effect.YAWN, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3, Effect.CURSE, Effect.ATTRACT,
    Effect.DISABLE, Effect.TORMENT, Effect.MAGNET_RISE, Effect.FOCUS_ENERGY, Effect.INGRAIN,
    Effect.AQUA_RING, Effect.PARTIALLY_TRAPPED, Effect.BIND, Effect.WRAP, Effect.FIRE_SPIN,
    Effect.WHIRLPOOL, Effect.SAND_TOMB, Effect.CLAMP, Effect.SLOW_START, Effect.FLASH_FIRE,
    Effect.CHARGE, Effect.STOCKPILE, Effect.DESTINY_BOND, Effect.LOCKED_MOVE, Effect.TYPECHANGE,
    Effect.HEAL_BLOCK, Effect.EMBARGO, Effect.IMPRISON, Effect.NIGHTMARE, Effect.UPROAR,
    Effect.SMACK_DOWN, Effect.THROAT_CHOP, Effect.SALT_CURE, Effect.NO_RETREAT,
    Effect.GLAIVE_RUSH, Effect.ROOST, Effect.TRAPPED, Effect.FLINCH, Effect.GASTRO_ACID,
    Effect.MINIMIZE, Effect.FORESIGHT, Effect.MIRACLE_EYE, Effect.POWER_TRICK,
]
WEATHERS = [w for w in Weather if w is not Weather.UNKNOWN]
FIELDS = [f for f in Field if f is not Field.UNKNOWN]
SCREENS = [
    SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL,
    SideCondition.SAFEGUARD, SideCondition.MIST, SideCondition.TAILWIND, SideCondition.LUCKY_CHANT,
]
HAZARDS = {  # condition -> max layers
    SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 3,
    SideCondition.TOXIC_SPIKES: 2, SideCondition.STICKY_WEB: 1,
}
CATEGORIES = [MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS]

N_MOVES, MOVEPOOL = 4, 24
N_CAT = 3 + N_MOVES  # species, item, ability, 4 moves
MOVE_DIM = 3 + 1 + 1 + len(CATEGORIES) + len(TYPES) + 18
MON_NUM_DIM = (
    2 + 6 + 6 + 1 + 2 * len(TYPES) + 1 + len(BOOSTS) + len(STATUSES) + 1
    + len(EFFECTS) + 10
)
MON_DIM = N_CAT + MOVEPOOL + MON_NUM_DIM + N_MOVES * MOVE_DIM
FIELD_DIM = 2 * len(WEATHERS) + 1 + 2 * len(FIELDS) + 2 * (2 * len(SCREENS) + len(HAZARDS)) + 14
N_TOKENS = 1 + 12
ACTIVE_FLAG = MON_NUM_DIM - 10  # index of mon.active inside the mon numerics block
OBS_DIM = FIELD_DIM + 12 * MON_DIM


@cache
def randbats_sets(battle_format: str) -> dict:
    path = DATA / f"{battle_format}_sets.json"
    return json.loads(path.read_text()) if path.exists() else {}


def idx(kind: str, key: str | None) -> int:
    if key is None:
        return UNK
    if key == "":
        return PAD
    table = INDEX[kind]
    if key in table:
        return table[key]
    if kind == "moves" and key.startswith("hiddenpower"):
        return table["hiddenpower"]
    return table.get(key.rstrip("0123456789"), UNK)


def one_hot(value, options) -> list[float]:
    return [float(value == o) for o in options]


def elapsed(start_turn: int, turn: int, horizon: float = 8.0) -> float:
    return min(max(turn - start_turn, 0), horizon) / horizon


def estimate_stat(base: int, level: int, stat: str) -> int:
    # Random battles: 31 IVs, 84 EVs, neutral nature.
    core = (2 * base + 31 + 21) * level // 100
    return core + level + 10 if stat == "hp" else core + 5


def type_multiplier(move: Move, defender: Pokemon | None, gen: int) -> float:
    if defender is None or move.category is MoveCategory.STATUS:
        return 1.0
    return move.type.damage_multiplier(
        defender.type_1, defender.type_2, type_chart=GenData.from_gen(gen).type_chart
    )


def encode_move(move: Move | None, user: Pokemon, defender: Pokemon | None, gen: int,
                available: set[str]) -> list[float]:
    if move is None:
        return [0.0] * MOVE_DIM
    mult = type_multiplier(move, defender, gen)
    acc = 1.0 if move.accuracy is True else float(move.accuracy)
    feats = [
        move.base_power / 150, acc, move.priority / 5,
        move.current_pp / max(move.max_pp, 1),
        float(move.id in available),
        *one_hot(move.category, CATEGORIES),
        *one_hot(move.type, TYPES),
        float(move.type in user.types),
        math.log2(mult) / 2 if mult > 0 else -1.5,
        float(mult == 0),
        move.heal, move.drain, move.recoil,
        float(bool(move.self_switch)), float(move.force_switch),
        float(move.side_condition is not None), float(move.status is not None),
        float(move.volatile_status is not None),
        sum((move.boosts or {}).values()) / 6,
        sum((move.self_boost or {}).values()) / 6,
        float(move.is_protect_move), float(move.breaks_protect),
        move.expected_hits / 5, float(move.weather is not None or move.terrain is not None),
        float(bool(move.self_destruct)),
    ]
    assert len(feats) == MOVE_DIM, len(feats)
    return feats


def encode_mon(mon: Pokemon | None, battle: AbstractBattle, mine: bool) -> np.ndarray:
    out = np.zeros(MON_DIM, dtype=np.float32)
    if mon is None:  # never-seen opponent slot
        out[N_CAT + MOVEPOOL + MON_NUM_DIM - 1] = 1.0  # unknown flag
        return out
    gen = battle.gen
    defender = battle.opponent_active_pokemon if mine else battle.active_pokemon
    moves = list(mon.moves.values())[:N_MOVES]
    moves += [None] * (N_MOVES - len(moves))
    available = {m.id for m in battle.available_moves} if mine and mon.active else set()

    ability = mon.ability
    if ability is None and len(mon.possible_abilities) == 1:
        ability = mon.possible_abilities[0]
    item = None if mon.item == GenData.UNKNOWN_ITEM else mon.item
    species = mon.species if mon.species in INDEX["species"] else mon.base_species
    cats = [idx("species", species), idx("items", item), idx("abilities", ability)]
    cats += [idx("moves", m.id) if m else PAD for m in moves]

    prior = randbats_sets(battle.format or "").get(species, {})
    pool = [m.id for m in mon.moves.values()] if mine else prior.get("moves", [])
    pool_ids = [idx("moves", m) for m in pool[:MOVEPOOL]]
    pool_ids += [PAD] * (MOVEPOOL - len(pool_ids))

    level = mon.level or prior.get("level", 80)
    stats = [
        (mon.stats.get(s) if mon.stats.get(s) else estimate_stat(mon.base_stats[s], level, s)) / 500
        for s in STATS
    ]
    boosts = [mon.boosts.get(b, 0) / 6 for b in BOOSTS]
    effects = [float(e in mon.effects) for e in EFFECTS]
    is_opp_active = battle.opponent_active_pokemon is mon
    nums = [
        mon.current_hp_fraction, float(mon.fainted),
        *[mon.base_stats[s] / 255 for s in STATS],
        *stats,
        level / 100,
        *[float(t in mon.types) for t in TYPES],
        *one_hot(mon.tera_type, TYPES),
        float(mon.is_terastallized),
        *boosts,
        *one_hot(mon.status, STATUSES),
        min(mon.status_counter, 10) / 10,
        *effects,
        float(mon.active), float(mine), float(mon.first_turn), mon.protect_counter / 3,
        float(mon.must_recharge), float(bool(mon.preparing)), math.log10(max(mon.weight, 0.1)) / 3,
        float(mine and mon.active and battle.trapped),
        float(is_opp_active), 0.0,  # last slot: unknown flag
    ]
    assert len(nums) == MON_NUM_DIM, len(nums)
    move_feats = [f for m in moves for f in encode_move(m, mon, defender, gen, available)]
    out[:] = cats + pool_ids + nums + move_feats
    return out


def encode_side(conditions: dict, turn: int) -> list[float]:
    screens = [float(c in conditions) for c in SCREENS]
    screen_age = [elapsed(conditions[c], turn) if c in conditions else 0.0 for c in SCREENS]
    hazards = [min(conditions.get(c, 0), n) / n for c, n in HAZARDS.items()]
    return screens + screen_age + hazards


def encode_field(battle: AbstractBattle) -> np.ndarray:
    turn = battle.turn
    opp_seen = len(battle.opponent_team)
    feats = [
        *[float(w in battle.weather) for w in WEATHERS],
        *[elapsed(battle.weather[w], turn) if w in battle.weather else 0.0 for w in WEATHERS],
        float(not battle.weather),
        *[float(f in battle.fields) for f in FIELDS],
        *[elapsed(battle.fields[f], turn) if f in battle.fields else 0.0 for f in FIELDS],
        *encode_side(battle.side_conditions, turn),
        *encode_side(battle.opponent_side_conditions, turn),
        min(turn, 100) / 100,
        (6 - opp_seen) / 6,
        sum(not m.fainted for m in battle.team.values()) / 6,
        (6 - sum(m.fainted for m in battle.opponent_team.values())) / 6,
        float(any(battle.force_switch) if isinstance(battle.force_switch, list) else battle.force_switch),
        float(battle.trapped), float(battle.maybe_trapped),
        float(battle.can_tera), float(battle.used_tera), float(battle.opponent_used_tera),
        float(battle.wait), float(battle.active_pokemon is None),
        len(battle.available_switches) / 5, len(battle.available_moves) / 4,
    ]
    assert len(feats) == FIELD_DIM, len(feats)
    return np.asarray(feats, dtype=np.float32)


def embed_battle(battle: AbstractBattle) -> np.ndarray:
    own = list(battle.team.values())[:6]
    own += [None] * (6 - len(own))
    opp = list(battle.opponent_team.values())[:6]
    opp += [None] * (6 - len(opp))
    return np.concatenate(
        [encode_field(battle)]
        + [encode_mon(m, battle, mine=True) for m in own]
        + [encode_mon(m, battle, mine=False) for m in opp]
    )
