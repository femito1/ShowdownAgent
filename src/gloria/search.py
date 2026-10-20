"""Decision-time search on top of the PPO network (Wang 2024, ch. 5 style), using poke-engine.

Each decision we
  1. sample `samples` determinizations of the battle: our side is fully known; the opponent's
     unrevealed moves / item / ability / mons are drawn from the gen4 random-battle sets,
     consistent with what has been revealed, and stats follow the random-battle rules
     (level from the data, 85 EVs, 31 IVs, neutral nature);
  2. run poke-engine's simultaneous-move MCTS on every sample for `ms` milliseconds (in parallel
     worker processes, `gloria.engine_worker`);
  3. score each legal action as  mean visit share over samples + prior_weight * network prob,
     optionally only among the network's top-k actions, and play the argmax.
Any failure (conversion, engine panic, unmappable choice) falls back to the network's argmax.

Approximations / ignored (documented, not bugs): opponent PP = what poke-env tracks; hidden power
is always 70 BP and its type for the opponent comes from the sampled set; the random-battle item
logic is a simplified re-implementation; volatile durations, wish, future sight, rest turns, the
opponent's choice lock and toxic counter, and partial-trap / disable / encore timers are not
transferred; substitute HP is assumed to be 1/4 max HP; weather from abilities is permanent
(gen4), other weather lasts 5 turns from when we first saw it; unrevealed opponent mons are
uniformly sampled species (no team-generation constraints).

    python -m gloria.search runs/<run>/best_model.zip --opponent policy --n-battles 100 --ms 300
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import poke_engine as pe
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, Battle, Effect, Field, Pokemon, PokemonType, SideCondition, Status, Weather
from poke_env.data import GenData
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from gloria.encoder import embed_battle
from gloria.engine_worker import EnginePool
from gloria.env import PolicyPlayer, local_server
from gloria.evaluate import BASELINES

log = logging.getLogger("gloria.search")
GEN = GenData.from_gen(4)
SETS_PATH = Path(__file__).resolve().parents[2] / "pokemon-showdown/data/random-battles/gen4/sets.json"
UNION_PATH = Path(__file__).parent / "data/gen4randombattle_sets.json"

STATUS = {Status.BRN: "burn", Status.SLP: "sleep", Status.FRZ: "freeze", Status.PAR: "paralyze",
          Status.PSN: "poison", Status.TOX: "toxic"}
WEATHER = {Weather.SUNNYDAY: "sun", Weather.RAINDANCE: "rain", Weather.SANDSTORM: "sand", Weather.HAIL: "hail"}
WEATHER_ABILITY = {Weather.SUNNYDAY: "drought", Weather.RAINDANCE: "drizzle",
                   Weather.SANDSTORM: "sandstream", Weather.HAIL: "snowwarning"}
SCREEN_TURNS = {SideCondition.REFLECT: ("reflect", 5), SideCondition.LIGHT_SCREEN: ("light_screen", 5),
                SideCondition.SAFEGUARD: ("safeguard", 5), SideCondition.MIST: ("mist", 5),
                SideCondition.LUCKY_CHANT: ("lucky_chant", 5), SideCondition.TAILWIND: ("tailwind", 3)}
SKIP_EFFECTS = {Effect.FLINCH, Effect.PROTECT, Effect.ENDURE}  # per-turn effects poke-env may still hold
EV, IV = 85, 31


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------- random-battle set data

@cache
def role_sets() -> dict[str, list[dict[str, Any]]]:
    """species -> [{role, movepool, abilities}] from the Showdown checkout, else the movepool union."""
    if SETS_PATH.exists():
        raw = json.loads(SETS_PATH.read_text())
        return {sp: [{"role": s["role"], "movepool": s["movepool"], "abilities": [norm(a) for a in s["abilities"]]}
                     for s in v["sets"]] for sp, v in raw.items()}
    union = json.loads(UNION_PATH.read_text())
    return {sp: [{"role": "", "movepool": v["moves"], "abilities": v["abilities"]}] for sp, v in union.items()}


@cache
def levels() -> dict[str, int]:
    return {sp: v["level"] for sp, v in json.loads(UNION_PATH.read_text()).items()}


def move_data(move: str) -> dict[str, Any]:
    return GEN.moves.get("hiddenpower" if move.startswith("hiddenpower") else move, {})


def matches(revealed: str, pool_move: str) -> bool:
    # poke-env reports the opponent's hidden power without its type
    return pool_move == revealed or (revealed == "hiddenpower" and pool_move.startswith("hiddenpower"))


def guess_item(species: str, role: str, moves: list[str], ability: str, rng: random.Random) -> str:
    """Simplified version of the gen4 random-battle item choice (pokemon-showdown teams.ts)."""
    dex = GEN.pokedex[species]
    spe, atk, spa = dex["baseStats"]["spe"], dex["baseStats"]["atk"], dex["baseStats"]["spa"]
    cats = [move_data(m).get("category") for m in moves]
    phys, spec = cats.count("Physical"), cats.count("Special")
    has = set(moves).__contains__
    priority = any(move_data(m).get("priority", 0) > 0 and move_data(m).get("category") != "Status" for m in moves)
    scarf_ok = role != "Wallbreaker" and 60 <= spe <= 108 and not priority and not has("pursuit")
    fixed = {"latias": "souldew", "latios": "souldew", "marowak": "thickclub", "pikachu": "lightball",
             "shedinja": "focussash", "smeargle": "focussash", "wobbuffet": "custapberry",
             "shuckle": "leftovers", "honchkrow": "lifeorb", "palkia": "lustrousorb",
             "giratinaorigin": "griseousorb", "farfetchd": "stick"}
    if species in fixed:
        return fixed[species]
    if ability == "poisonheal" or has("facade"):
        return "toxicorb"
    if has("trick") or has("switcheroo") or has("healingwish"):
        return "choicescarf" if scarf_ok else "choiceband" if phys > spec else "choicespecs"
    if has("bellydrum") or ability == "unburden":
        return "sitrusberry"
    if has("waterspout"):
        return "choicescarf"
    if ability == "magicguard":
        return "lifeorb"
    if has("reflect") and has("lightscreen"):
        return "lightclay"
    if has("rest") and not has("sleeptalk") and ability not in ("naturalcure", "shedskin"):
        return "chestoberry"
    if role == "Staller":
        item = "leftovers"
    elif spec == 4:
        item = "choicescarf" if scarf_ok and spa >= 90 and rng.random() < 0.5 else "choicespecs"
    elif phys == 4 and not (has("fakeout") or has("rapidspin")):
        item = "choicescarf" if scarf_ok and atk >= 100 and rng.random() < 0.5 else "choiceband"
    elif has("protect") or has("substitute"):
        item = "leftovers"
    elif role == "Fast Support":
        item = "lifeorb" if phys + spec >= 3 and not (has("rapidspin") or has("uturn")) else "leftovers"
    elif role in ("Fast Attacker", "Setup Sweeper", "Wallbreaker") and not has("rapidspin"):
        item = "lifeorb"
    else:
        item = "leftovers"
    return "blacksludge" if item == "leftovers" and "Poison" in dex["types"] else item


def sample_set(species: str, revealed_moves: list[str], ability: str | None,
               rng: random.Random) -> tuple[str, list[str], str]:
    """(role, 4 moves, ability) of a random-battle set consistent with what was revealed."""
    sets = role_sets().get(species) or [{"role": "", "movepool": revealed_moves or ["tackle"], "abilities": []}]
    ok = [s for s in sets
          if all(any(matches(r, m) for m in s["movepool"]) for r in revealed_moves)
          and (ability is None or not s["abilities"] or ability in s["abilities"])]
    chosen = rng.choice(ok or sets)
    pool = chosen["movepool"]
    moves = [next((m for m in pool if matches(r, m)), r) for r in revealed_moves][:4]
    rest = [m for m in pool if m not in moves]
    rng.shuffle(rest)
    for m in rest:
        if len(moves) == 4:
            break
        if m.startswith("hiddenpower") and any(x.startswith("hiddenpower") for x in moves):
            continue
        moves.append(m)
    if ability is None:
        ability = rng.choice(chosen["abilities"]) if chosen["abilities"] else "none"
    return chosen["role"], moves, ability


# ---------------------------------------------------------------- poke-env -> poke-engine

def engine_move(move_id: str, hp_type: PokemonType | None = None) -> str:
    """poke-engine id: gen4 hidden power is a typed 70 BP move ('hiddenpowerfire70')."""
    if move_id.startswith("hiddenpower"):
        typed = move_id[len("hiddenpower"):].rstrip("0123456789")
        if not typed and hp_type is not None:
            typed = hp_type.name.lower()
        return f"hiddenpower{typed}70" if typed else "hiddenpower"
    return move_id


def engine_types(mon: Pokemon) -> tuple[str, str]:
    ts = [t.name.lower() for t in mon.types if t not in (None, PokemonType.THREE_QUESTION_MARKS)]
    ts = (ts + ["typeless", "typeless"])[:2]
    return ts[0], ts[1]


def calc_stats(species: str, level: int, physical: bool) -> dict[str, int]:
    base = GEN.pokedex[species]["baseStats"]
    out = {"hp": (2 * base["hp"] + IV + EV // 4) * level // 100 + level + 10}
    for s in ("atk", "def", "spa", "spd", "spe"):
        ev, iv = (0, 0) if s == "atk" and not physical else (EV, IV)  # random battles minimise confusion damage
        out[s] = (2 * base[s] + iv + ev // 4) * level // 100 + 5
    return out


def make_mon(species: str, level: int, types: tuple[str, str], hp: int, maxhp: int, stats: dict[str, int],
             ability: str, item: str, moves: list[tuple[str, int, bool]], status: str, sleep_turns: int,
             weight: float) -> pe.Pokemon:
    return pe.Pokemon(
        id=species, level=level, types=types, base_types=types, hp=hp, maxhp=maxhp,
        ability=ability or "none", base_ability=ability or "none", item=item or "none",
        attack=stats["atk"], defense=stats["def"], special_attack=stats["spa"],
        special_defense=stats["spd"], speed=stats["spe"], status=status, sleep_turns=sleep_turns,
        weight_kg=float(weight), moves=[pe.Move(id=m, pp=max(pp, 0), disabled=d) for m, pp, d in moves],
    )


def status_of(mon: Pokemon) -> tuple[str, int]:
    status = STATUS.get(mon.status, "none") if mon.status is not None else "none"
    return status, mon.status_counter if status == "sleep" else 0


def own_mon(mon: Pokemon, disabled: set[str]) -> tuple[pe.Pokemon, list[str]]:
    """Our mon exactly as the request describes it; also returns its engine move ids (action order)."""
    moves = list(mon.moves.values())[:4]
    ids = [norm(engine_move(m.id, m.type)) for m in moves]
    status, sleep = status_of(mon)
    hp = 0 if mon.fainted else mon.current_hp
    item = mon.item if mon.item and mon.item != GEN.UNKNOWN_ITEM else "none"
    stats = {k: mon.stats.get(k) or 100 for k in ("atk", "def", "spa", "spd", "spe")}
    mv = [(i, m.current_pp, m.id in disabled) for i, m in zip(ids, moves)]
    return make_mon(mon.species, mon.level, engine_types(mon), hp, mon.max_hp or 1, stats, mon.ability or "none",
                    item, mv, status, sleep, mon.weight), ids


def opp_mon(mon: Pokemon | None, species: str, rng: random.Random) -> pe.Pokemon:
    """Opponent mon (None = unrevealed, sampled from `species`) with hidden info determinized."""
    revealed = [m for m in (mon.moves if mon else {}) if norm(m) != "struggle"]
    role, moves, ability = sample_set(species, revealed, mon.ability if mon else None, rng)
    item = mon.item if mon and mon.item != GEN.UNKNOWN_ITEM else None
    if item is None:
        item = guess_item(species, role, moves, ability, rng)
    level = mon.level if mon else levels().get(species, 80)
    physical = any(move_data(m).get("category") == "Physical" for m in moves)
    stats = calc_stats(species, level, physical)
    pp = {m: mv.current_pp for m, mv in mon.moves.items()} if mon else {}
    mv = [(norm(engine_move(m)), pp.get("hiddenpower" if m.startswith("hiddenpower") else m, 16), False) for m in moves]
    status, sleep = status_of(mon) if mon else ("none", 0)
    frac = 0.0 if mon and mon.fainted else mon.current_hp_fraction if mon else 1.0
    hp = 0 if frac <= 0 else max(1, round(frac * stats["hp"]))
    types = engine_types(mon) if mon else tuple((t.lower() for t in GEN.pokedex[species]["types"] + ["typeless"]))[:2]
    weight = mon.weight if mon else GEN.pokedex[species].get("weightkg", 50.0)
    return make_mon(species, level, types, hp, stats["hp"], stats, ability, item or "none", mv, status, sleep, weight)


def side_conditions(conds: dict[SideCondition, int], turn: int, toxic_count: int) -> pe.SideConditions:
    kw: dict[str, int] = {"toxic_count": toxic_count}
    kw["spikes"] = conds.get(SideCondition.SPIKES, 0)
    kw["toxic_spikes"] = conds.get(SideCondition.TOXIC_SPIKES, 0)
    kw["stealth_rock"] = int(SideCondition.STEALTH_ROCK in conds)
    for cond, (name, length) in SCREEN_TURNS.items():
        if cond in conds:
            kw[name] = max(1, length - (turn - conds[cond]))
    return pe.SideConditions(**kw)


def make_side(mons: list[pe.Pokemon], active: Pokemon | None, active_idx: int, conds: pe.SideConditions,
              last_move: int | None, **kw: Any) -> pe.Side:
    """`last_move`: index of the move the active mon is locked into, if known (encore needs it)."""
    mons = (mons + [pe.Pokemon.create_fainted() for _ in range(6)])[:6]
    boosts = active.boosts if active else {}
    effects = {e for e in (active.effects if active else {}) if e not in SKIP_EFFECTS}
    volatiles = {e.name.replace("_", "") for e in effects}
    if last_move is None:
        volatiles.discard("ENCORE")  # the engine panics on encore without a last-used move
    if active is not None and active.must_recharge:
        volatiles.add("MUSTRECHARGE")
    sub = mons[active_idx].maxhp // 4 if Effect.SUBSTITUTE in effects else 0
    return pe.Side(
        pokemon=mons, active_index=str(active_idx), side_conditions=conds, volatile_statuses=volatiles,
        substitute_health=sub, attack_boost=boosts.get("atk", 0), defense_boost=boosts.get("def", 0),
        special_attack_boost=boosts.get("spa", 0), special_defense_boost=boosts.get("spd", 0),
        speed_boost=boosts.get("spe", 0), accuracy_boost=boosts.get("accuracy", 0),
        evasion_boost=boosts.get("evasion", 0),
        last_used_move="move:none" if last_move is None else f"move:{last_move}", **kw,
    )


def weather_turns(battle: AbstractBattle, weather: Weather, first_seen: int) -> int:
    """Turns left; -1 (never decremented by the engine) for gen4's permanent ability weather."""
    mons = list(battle.team.values()) + list(battle.opponent_team.values())
    ability = WEATHER_ABILITY.get(weather)
    if any(m.ability == ability or (m.ability is None and ability in m.possible_abilities) for m in mons):
        return -1
    return max(1, 5 - (battle.turn - first_seen))


class Converted:
    """One determinization plus the maps from engine choices back to poke-env action indices."""

    def __init__(self, state: pe.State, switch_ids: list[str], move_ids: list[str]):
        self.state_str = state.to_string()
        self.switch_ids, self.move_ids = switch_ids, move_ids

    def action(self, choice: str) -> int | None:
        if choice.startswith("switch "):
            name = choice[len("switch "):]
            return self.switch_ids.index(name) if name in self.switch_ids else None
        return 6 + self.move_ids.index(choice) if choice in self.move_ids else None


def convert(battle: Battle, rng: random.Random, weather_seen: int | None = None) -> Converted:
    team = list(battle.team.values())
    active_idx = next((i for i, m in enumerate(team) if m.active), 0)
    disabled = set()
    if battle.active_pokemon is not None and not battle.force_switch:
        avail = {m.id for m in battle.available_moves}
        disabled = {m for m in battle.active_pokemon.moves if m not in avail}
    ours, move_ids = [], []
    for i, mon in enumerate(team):
        m, ids = own_mon(mon, disabled if i == active_idx else set())
        ours.append(m)
        if i == active_idx:
            move_ids = ids
    active = team[active_idx]
    tox = active.status_counter if active.status is Status.TOX else 0
    avail = [m.id for m in battle.available_moves]
    locked = list(active.moves).index(avail[0]) if len(avail) == 1 and avail[0] in list(active.moves)[:4] else None
    side_one = make_side(ours, active, active_idx, side_conditions(battle.side_conditions, battle.turn, tox),
                         locked, force_switch=bool(battle.force_switch), force_trapped=bool(battle.trapped))

    opp = list(battle.opponent_team.values())
    known = {m.species for m in opp}
    pool = [s for s in role_sets() if s not in known and s in GEN.pokedex]
    unrevealed = rng.sample(pool, max(0, min(6, battle.max_team_size or 6) - len(opp)))
    theirs = [opp_mon(m, m.species, rng) for m in opp] + [opp_mon(None, s, rng) for s in unrevealed]
    opp_active = battle.opponent_active_pokemon
    opp_idx = next((i for i, m in enumerate(opp) if m is opp_active), 0)
    side_two = make_side(theirs, opp_active, opp_idx, side_conditions(battle.opponent_side_conditions, battle.turn, 0),
                         None)

    weather, turns = "none", 0
    for w in battle.weather:
        if w in WEATHER:
            weather, turns = WEATHER[w], weather_turns(battle, w, battle.turn if weather_seen is None else weather_seen)
    tr = Field.TRICK_ROOM in battle.fields
    tr_turns = max(1, 5 - (battle.turn - battle.fields[Field.TRICK_ROOM])) if tr else 0
    state = pe.State(side_one=side_one, side_two=side_two, weather=weather, weather_turns_remaining=turns,
                     trick_room=tr, trick_room_turns_remaining=tr_turns)
    return Converted(state, [norm(m.species) for m in team], move_ids)


# ---------------------------------------------------------------- player

_POOLS: dict[int, EnginePool] = {}


def engine_pool(workers: int | None = None) -> EnginePool:
    """Process-wide pool of poke-engine worker subprocesses, shared by all SearchPlayers."""
    n = workers or max(1, (os.cpu_count() or 2) - 1)
    return _POOLS.setdefault(n, EnginePool(n))


class SearchPlayer(Player):
    """Network-guided determinized MCTS. prior_weight=0 and top_k=None is pure engine search."""

    def __init__(self, policy: ActorCriticPolicy, *args: Any, ms: int = 300, samples: int = 4,
                 prior_weight: float = 0.5, top_k: int | None = None, workers: int | None = None,
                 seed: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy, self.ms, self.samples = policy, ms, samples
        self.prior_weight, self.top_k = prior_weight, top_k
        self.pool = engine_pool(workers)
        self.rng = random.Random(seed)
        self.weather_seen: dict[str, tuple[Weather, int]] = {}
        self.stats = {"decisions": 0, "searched": 0, "fallbacks": 0, "agree": 0, "failed_samples": 0}

    def network_probs(self, battle: Battle) -> tuple[np.ndarray, np.ndarray]:
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.float32)
        obs = {"observation": torch.as_tensor(embed_battle(battle)).unsqueeze(0),
               "action_mask": torch.as_tensor(mask).unsqueeze(0)}
        obs = {k: v.to(self.policy.device) for k, v in obs.items()}
        with torch.no_grad():
            probs = self.policy.get_distribution(obs).distribution.probs[0].cpu().numpy()
        return probs, mask

    def _weather_first_seen(self, battle: Battle) -> int | None:
        current = next(iter(battle.weather), None)
        seen = self.weather_seen.get(battle.battle_tag)
        if current is None:
            self.weather_seen.pop(battle.battle_tag, None)
            return None
        if seen is None or seen[0] != current:
            # weather started during the previous turn (or this one, on a switch-in)
            seen = self.weather_seen[battle.battle_tag] = (current, max(battle.turn - 1, 0))
        return seen[1]

    async def visit_shares(self, battle: Battle) -> np.ndarray:
        """Mean (over determinizations) MCTS visit share of every poke-env action index."""
        first_seen = self._weather_first_seen(battle)
        convs = [convert(battle, self.rng, first_seen) for _ in range(self.samples)]
        results = await asyncio.gather(*(self.pool.search(c.state_str, self.ms) for c in convs),
                                       return_exceptions=True)
        shares, ok, errors = np.zeros(10), 0, []
        for conv, options in zip(convs, results):  # a failed sample (engine panic) is just skipped
            try:
                shares += self._sample_shares(conv, options)
                ok += 1
            except Exception as e:
                errors.append(e)
        self.stats["failed_samples"] += len(errors)
        if not ok:
            raise errors[0]
        return shares / ok

    @staticmethod
    def _sample_shares(conv: Converted, options: list | BaseException) -> np.ndarray:
        if isinstance(options, BaseException):
            raise options
        shares, total = np.zeros(10), sum(v for _, v, _ in options) or 1
        for choice, visits, _ in options:
            a = conv.action(choice)
            if a is None:
                raise ValueError(f"unmappable engine choice {choice!r} ({conv.switch_ids}, {conv.move_ids})")
            shares[a] += visits / total
        return shares

    def combine(self, shares: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> int:
        legal = mask > 0
        if self.top_k:
            legal &= probs >= np.sort(np.where(legal, probs, -1))[::-1][min(self.top_k, int(legal.sum())) - 1]
        score = np.where(legal, shares + self.prior_weight * probs, -np.inf)
        return int(np.argmax(score))

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        assert isinstance(battle, Battle)
        probs, mask = self.network_probs(battle)
        net_action = int(np.argmax(np.where(mask > 0, probs, -1)))
        action = net_action
        self.stats["decisions"] += 1
        if mask.sum() > 1:
            try:
                shares = await self.visit_shares(battle)
                if shares[mask > 0].sum() < 0.5:  # the engine searched mostly illegal options -> desync
                    raise ValueError(f"engine options disagree with the request: {shares} vs mask {mask}")
                action = self.combine(shares, probs, mask)
                self.stats["searched"] += 1
                self.stats["agree"] += action == net_action
            except Exception as e:
                self.stats["fallbacks"] += 1
                log.warning("search failed in %s turn %d, using the network: %s", battle.battle_tag, battle.turn, e)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self.weather_seen.pop(battle.battle_tag, None)


# ---------------------------------------------------------------- CLI

async def run(args: argparse.Namespace) -> None:
    policy = PPO.load(args.model, device="cpu").policy.eval()
    tag = random.randint(0, 10**6)
    common = dict(battle_format=args.format, server_configuration=local_server(args.port),
                  max_concurrent_battles=args.concurrency, log_level=40)
    player = SearchPlayer(policy, ms=args.ms, samples=args.samples, prior_weight=args.prior_weight,
                          top_k=args.top_k, workers=args.workers,
                          account_configuration=AccountConfiguration(f"search{tag}", None), **common)
    opp_conf = AccountConfiguration(f"{args.opponent[:8]}{tag}", None)
    opponent = (PolicyPlayer(policy, account_configuration=opp_conf, **common) if args.opponent == "policy"
                else BASELINES[args.opponent](account_configuration=opp_conf, **common))
    start = time.time()
    await player.battle_against(opponent, n_battles=args.n_battles)
    n, won = player.n_finished_battles, player.n_won_battles
    s = player.stats
    print(f"SearchPlayer(ms={args.ms}, samples={args.samples}, prior_weight={args.prior_weight}, top_k={args.top_k})"
          f" vs {args.opponent}: won {won}/{n} = {won / max(n, 1):.1%}  "
          f"(±{1.96 * np.sqrt(max(won, 1) * max(n - won, 1) / max(n, 1) ** 3):.1%})")
    print(f"decisions {s['decisions']}, searched {s['searched']}, fallbacks {s['fallbacks']}, "
          f"failed samples {s['failed_samples']}, "
          f"same as network argmax {s['agree'] / max(s['searched'], 1):.1%}, {time.time() - start:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--opponent", default="policy", choices=["policy", *BASELINES])
    p.add_argument("--n-battles", type=int, default=50)
    p.add_argument("--ms", type=int, default=300, help="MCTS time per determinization per decision")
    p.add_argument("--samples", type=int, default=4, help="determinizations per decision")
    p.add_argument("--prior-weight", type=float, default=0.5, help="0 = pure engine search")
    p.add_argument("--top-k", type=int, default=None, help="only consider the network's top-k actions")
    p.add_argument("--concurrency", type=int, default=3, help="simultaneous battles")
    p.add_argument("--workers", type=int, default=None, help="engine processes (default: cores - 1)")
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
