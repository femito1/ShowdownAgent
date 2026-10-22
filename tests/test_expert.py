"""Expert-iteration targets: shaped state values, search targets, shard round-trip."""

import logging

import numpy as np
import pytest
from poke_env.battle import Battle

pytest.importorskip("poke_engine")
from porygon.encoder import OBS_DIM  # noqa: E402
from porygon.expert import ShardWriter, load_shards, search_target, state_value  # noqa: E402

START = """|player|p1|me|
|player|p2|you|
|teamsize|p1|6
|teamsize|p2|6
|switch|p1a: Blissey|Blissey, L84, F|100/100
|switch|p2a: Jirachi|Jirachi, L80|100/100"""


def battle_from(log: str) -> Battle:
    battle = Battle("battle-gen4randombattle-1", "me", logging.getLogger("test"), gen=4)
    battle.player_role = "p1"
    for line in (START + "\n" + log).splitlines():
        battle.parse_message(line.split("|"))
    return battle


def test_state_value_matches_training_reward():
    assert state_value(battle_from("|turn|1")) == pytest.approx(0.0)
    hurt = battle_from("|turn|1\n|-damage|p2a: Jirachi|50/100\n|turn|2")
    assert state_value(hurt) == pytest.approx(0.05 * 0.5)
    fainted = battle_from("|turn|1\n|faint|p2a: Jirachi\n|turn|2")
    assert state_value(fainted) == pytest.approx(0.05 + 0.05)
    for winner, value in (("me", 1.0), ("you", -1.0)):
        battle = battle_from("|turn|1")
        battle.won_by(winner)
        assert state_value(battle) == pytest.approx(value)


def test_search_target_is_legal_and_normalised():
    mask = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0], dtype=np.float32)
    shares = np.array([0.3, 0.1, 0, 0, 0, 0, 0.6, 0, 0, 0])
    probs = np.full(10, 0.1)
    t = search_target(shares, probs, mask, prior_weight=0.25)
    assert t.sum() == pytest.approx(1.0) and (t[mask == 0] == 0).all()
    assert t.argmax() == 6


def test_shard_round_trip(tmp_path):
    w = ShardWriter(tmp_path, games_per_shard=2)
    rec = lambda r: {"obs": np.zeros(OBS_DIM, np.float16), "mask": np.ones(10, np.int8),  # noqa: E731
                     "prior": np.full(10, 0.1, np.float32), "target": np.full(10, 0.1, np.float32),
                     "ret": np.float32(r)}
    w.add_game([rec(1.0), rec(0.5)])
    w.add_game([rec(-1.0)])  # second game -> shard written
    w.add_game([rec(0.25)])
    w.flush()
    assert w.shards == 2 and w.total_records == 4
    data = load_shards([tmp_path])
    assert data["obs"].shape == (4, OBS_DIM) and data["obs"].dtype == np.float16
    assert sorted(data["ret"].tolist()) == [-1.0, 0.25, 0.5, 1.0]
