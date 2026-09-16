"""Feature tests, centred on the one failure that cannot be recovered from.

A leak does not raise. It produces a model that grades beautifully and loses
money, and by the time that is visible the cause is weeks behind you. So the
leak tests here are built to FAIL if the shift is removed, and each one was
checked against a deliberately broken copy of the module rather than assumed
to work.

This project has written a leak test that could not fail twice already. Both
times it passed on a correct implementation and also on a broken one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cfb_features as F


def _history(n_weeks=6, n_players=4, seed=0):
    """A small league where every number is distinct and increasing.

    Made-up but structured: player p's points in week w are 10*p + w, so any
    feature that accidentally includes the current row is arithmetically
    obvious rather than a matter of judgement.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        for p in range(n_players):
            team = "A" if p < n_players // 2 else "B"
            opp = "B" if team == "A" else "A"
            rows.append({
                "athlete_id": f"p{p}", "name": f"Player {p}",
                "season": 2025, "week": w, "school": team, "opponent": opp,
                "is_home": 1 if team == "A" else 0,
                "position": ["QB", "RB", "WR", "TE"][p % 4],
                "carries": float(rng.integers(0, 20)),
                "rec": float(rng.integers(0, 10)),
                "rush_yards": float(rng.integers(0, 120)),
                "rec_yards": float(rng.integers(0, 140)),
                "pass_yards": float(rng.integers(0, 300)),
                "completions": float(rng.integers(0, 25)),
                "points": 10.0 * p + w,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- the leak

def test_no_feature_contains_the_row_it_predicts():
    """Change one week's outcome; that week's own features must not move.

    This is the test that can fail. Tampering with a single player-week and
    re-deriving proves the direction of information flow: features for the
    tampered week must be byte-identical, and features for LATER weeks must
    change, because a shifted feature is still supposed to see the past.

    A test that only checked the first half would pass on a module that
    produced constant features, which is why the second half is not optional.
    """
    hist = _history()
    before = F.build(hist)

    tampered = hist.copy()
    mask = (tampered["athlete_id"] == "p1") & (tampered["week"] == 3)
    assert mask.sum() == 1
    tampered.loc[mask, "points"] = 999.0
    tampered.loc[mask, "carries"] = 99.0
    after = F.build(tampered)

    key = ["player_id", "season", "week"]
    b = before.sort_values(key).reset_index(drop=True)
    a = after.sort_values(key).reset_index(drop=True)

    same_week = (b["player_id"] == "p1") & (b["week"] == 3)
    later = (b["player_id"] == "p1") & (b["week"] > 3)

    # Only this player's OWN features are checked; team aggregates for week 3
    # legitimately change, because his team's week-3 total really did change.
    own = [f"ewm_{c}" for c in F.USAGE] + [
        "ewm_points", "ewm_touches", "sd_points", "played_last",
        "ewm_played", "played_rate"]
    for col in own:
        lhs = b.loc[same_week, col].to_numpy(dtype=float)
        rhs = a.loc[same_week, col].to_numpy(dtype=float)
        assert np.allclose(lhs, rhs, equal_nan=True), (
            f"{col} changed for the tampered week itself - it can see the "
            f"outcome it is meant to predict")

    moved = [c for c in own
             if not np.allclose(b.loc[later, c].to_numpy(dtype=float),
                                a.loc[later, c].to_numpy(dtype=float),
                                equal_nan=True)]
    assert moved, ("no later feature responded to the tampering, so this "
                   "test cannot detect a leak and proves nothing")


def test_first_game_has_no_history():
    """A player's debut must have empty history, not zero-filled history."""
    out = F.build(_history())
    first = out[out["week"] == 1]
    assert first["games_played"].eq(0).all()
    assert first["ewm_points"].isna().all(), "debut cannot have a prior mean"
    assert first["played_last"].isna().all()


def test_ewm_is_strictly_backward_looking():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    got = F._ewm(s, halflife=1e9)          # effectively a running mean
    assert np.isnan(got.iloc[0])
    assert abs(got.iloc[1] - 1.0) < 1e-9
    assert abs(got.iloc[2] - 1.5) < 1e-9
    assert abs(got.iloc[3] - 2.0) < 1e-9


# ------------------------------------------------------------------- shapes

def test_every_declared_feature_is_produced():
    out = F.build(_history())
    for c in F.FEATURES:
        assert c in out.columns, c


def test_shares_sum_to_one_within_a_team_game():
    out = F.build(_history())
    g = out.groupby(["team", "season", "week"])["share_carries"].sum()
    for v in g:
        assert abs(v - 1.0) < 1e-9 or v == 0.0, v


def test_home_flag_reads_numbers_as_numbers():
    """The NFL bug: float 1.0 -> "1.0" -> not "1" -> every row silently 0."""
    df = pd.DataFrame({"is_home": [1.0, 0.0, 1.0]})
    got = F._home_flag(df)
    assert list(got) == [1.0, 0.0, 1.0], list(got)
    assert got.sum() > 0, "a column of zeros looks exactly like real data"


def test_home_flag_reads_text_too():
    df = pd.DataFrame({"home_away": ["home", "away", "neutral"]})
    got = F._home_flag(df)
    assert got.iloc[0] == 1.0 and got.iloc[1] == 0.0


def test_home_flag_returns_missing_not_false_when_unknown():
    """'We do not know' and 'no' are different answers."""
    df = pd.DataFrame({"is_home": ["?", "??"]})
    assert F._home_flag(df).isna().all()


def test_margin_is_signed_and_opposite():
    hist = _history()
    df = F.to_model_frame(hist)
    df["touches"] = 0.0
    m = F._margins(df)
    for (s, w), g in m.groupby(["season", "week"]):
        assert abs(g["margin"].sum()) < 1e-9, "margins must cancel"


def test_trainable_excludes_debuts_and_non_skill_positions():
    hist = _history(n_weeks=8)
    out = F.build(hist)
    tr = F.trainable(out)
    assert tr["games_played"].min() >= F.MIN_PRIOR_GAMES
    assert set(tr["position"]) <= set(F.SKILL_POSITIONS)
    assert len(tr) < len(out), "trainable must actually exclude something"


def test_to_model_frame_refuses_a_frame_it_cannot_use():
    try:
        F.to_model_frame(pd.DataFrame({"nonsense": [1]}))
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("should have refused")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except Exception as exc:                   # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
