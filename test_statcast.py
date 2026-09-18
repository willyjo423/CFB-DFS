"""Statcast: the reduction, the join, and the leak it must not have.

Runs as a script: `python test_statcast.py`.

Two things are being tested and they fail in opposite directions.

The REDUCTION turns pitches into per-game rates. If it is wrong the feature is
merely useless, which a grade run would catch.

The JOIN rolls those rates forward into a feature. If THAT is wrong - if a
game's own Statcast reaches the row projecting that game - every backtest gets
better and the improvement is fiction. A leaked feature grades BETTER than the
truth, which is far harder to notice than the opposing-starter feature that
simply graded worse. So the leak test here matters more than the rest put
together, and it comes with a control that breaks the shift on purpose to
prove the test can fire.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import mlb_sport as MS
import mlb_statcast as MSC


def pitches(n_games: int = 6, seed: int = 0) -> pd.DataFrame:
    """A pitch-level frame shaped like Savant's, for two players."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        gpk = 700000 + g
        for _ in range(40):
            crushed = rng.random() < 0.5
            in_play = rng.random() < 0.6
            rows.append({
                "game_pk": gpk,
                "game_date": f"2026-05-{g + 1:02d}",
                "batter": 111, "pitcher": 222,
                "estimated_woba_using_speedangle": (
                    float(rng.uniform(0.3, 0.9)) if in_play else np.nan),
                "launch_speed": (float(rng.uniform(95, 110) if crushed
                                       else rng.uniform(60, 85))
                                 if in_play else np.nan),
                "launch_angle": float(rng.uniform(-20, 40)) if in_play else np.nan,
                "launch_speed_angle": (6 if (in_play and crushed) else 3)
                if in_play else np.nan,
                "description": "hit_into_play" if in_play else
                rng.choice(["ball", "swinging_strike", "foul",
                            "called_strike"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The reduction
# ---------------------------------------------------------------------------

def one_row_per_player_per_game_per_side():
    out = MSC.reduce_pitches(pitches(6))
    assert not out.empty
    assert set(out["side"]) == {"batter", "pitcher"}
    for side in ("batter", "pitcher"):
        s = out[out["side"] == side]
        assert len(s) == 6, f"{side}: expected 6 games, got {len(s)}"
        assert not s.duplicated(["game_pk", "player_id"]).any()


def counts_not_rates_so_chunks_can_be_added():
    """A week's rate cannot be averaged with another week's rate.

    The reduction must emit counts, so two chunks of the same game sum
    correctly. Rates are computed once, at the end, in fetch_one.
    """
    out = MSC.reduce_pitches(pitches(4))
    for c in ("bip", "barrel", "hard_hit", "swing", "whiff", "pitches"):
        assert c in out.columns, f"{c} missing - the reduction emitted rates"
    assert "sc_barrel_rate" not in out.columns, (
        "the reduction computed a rate; chunks can no longer be summed")


def a_barrel_is_the_league_definition():
    """launch_speed_angle == 6, not a re-derivation from speed and angle."""
    p = pitches(3)
    n_barrels = int((p["launch_speed_angle"] == 6).sum())
    out = MSC.reduce_pitches(p)
    got = float(out[out["side"] == "batter"]["barrel"].sum())
    assert got == n_barrels, f"counted {got} barrels, the feed has {n_barrels}"


def a_changed_feed_is_reported_not_guessed():
    """Missing columns must return empty and say so, not invent values."""
    p = pitches(3).drop(columns=["game_pk"])
    assert MSC.reduce_pitches(p).empty


def an_empty_fetch_is_not_a_crash():
    assert MSC.reduce_pitches(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# The join, and the leak
# ---------------------------------------------------------------------------

def history(days: int = 30, seed: int = 0) -> pd.DataFrame:
    from datetime import date, timedelta
    start = date(2026, 4, 1)
    rng = np.random.default_rng(seed)
    rows = []
    pos = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    for d in range(days):
        gpk = 800000 + d
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            for i in range(9):
                rows.append({
                    "player_id": f"H_{team}_{i}", "name": f"H {team} {i}",
                    "team": team, "opponent": opp,
                    "date": (start + timedelta(days=d)).isoformat(),
                    "game_pk": gpk, "is_pitcher": 0, "position": pos[i],
                    "plate_appearances": 4.0, "at_bats": 4.0,
                    "single": rng.poisson(0.9), "double": rng.poisson(0.15),
                    "triple": 0.0, "home_run": rng.poisson(0.12),
                    "rbi": rng.poisson(0.3), "run": rng.poisson(0.4),
                    "walk": rng.poisson(0.3), "stolen_base": 0.0,
                    "bat_slot": float(i + 1), "bat_started": 1.0,
                    "points": float(rng.normal(8.0, 3.0)),
                })
            rows.append({
                "player_id": f"SP_{team}", "name": f"SP {team}",
                "team": team, "opponent": opp,
                "date": (start + timedelta(days=d)).isoformat(),
                "game_pk": gpk, "is_pitcher": 1, "position": "SP",
                "innings": 6.0, "batters_faced": 24.0, "strikeout": 6.0,
                "hit_allowed": 5.0, "walk_allowed": 2.0, "earned_run": 3.0,
                "points": 12.0,
            })
    return pd.DataFrame(rows)


def statcast_for(hist: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in hist.itertuples(index=False):
        rows.append({
            "game_pk": str(r.game_pk),
            "player_id": str(r.player_id),
            "side": "pitcher" if r.is_pitcher else "batter",
            "sc_xwobacon": float(rng.uniform(0.25, 0.55)),
            "sc_barrel_rate": float(rng.uniform(0.0, 0.2)),
            "sc_hard_hit_rate": float(rng.uniform(0.2, 0.6)),
            "sc_whiff_rate": float(rng.uniform(0.1, 0.4)),
        })
    return pd.DataFrame(rows)


def the_feature_reaches_the_frame():
    hist = history()
    built = MS.build(hist, "hitters", statcast_for(hist))
    for c in ("sc_xwobacon", "ewm_sc_xwobacon", "ewm_sc_barrel_rate"):
        assert c in built.columns, f"{c} never arrived"
    have = float(built["sc_xwobacon"].notna().mean())
    assert have > 0.9, f"only {100 * have:.0f}% of rows matched"


def missing_statcast_is_a_supported_state():
    """No cache must mean the model is what it was, not a crash."""
    hist = history()
    built = MS.build(hist, "hitters", None)
    assert "ewm_sc_xwobacon" in built.columns
    assert built["ewm_sc_xwobacon"].isna().all()
    b2 = MS.build(hist, "hitters", pd.DataFrame())
    assert len(b2) == len(built)


def the_join_does_not_fan_out():
    hist = history()
    hitters, _ = MS.split(hist)
    built = MS.build(hist, "hitters", statcast_for(hist))
    assert len(built) == len(hitters), (
        f"{len(hitters)} rows became {len(built)}")


def a_game_cannot_see_its_own_statcast():
    """THE test. A leaked feature grades better than the truth.

    Change one game's Statcast and the rolled-forward value attached to THAT
    game must not move.
    """
    hist = history()
    sc = statcast_for(hist)
    a = MS.build(hist, "hitters", sc).sort_values(
        ["player_id", "game_pk"])["ewm_sc_xwobacon"].to_numpy()

    sc2 = sc.copy()
    target = sc2["game_pk"] == str(hist["game_pk"].max())
    sc2.loc[target, "sc_xwobacon"] = 9.99
    b = MS.build(hist, "hitters", sc2).sort_values(
        ["player_id", "game_pk"])["ewm_sc_xwobacon"].to_numpy()

    assert np.allclose(np.nan_to_num(a, nan=-1.0),
                       np.nan_to_num(b, nan=-1.0)), (
        "changing a game's Statcast changed the feature used to project that "
        "same game - this is a leak, and it would make every backtest look "
        "better than the truth")


def the_leak_test_can_fail():
    """The control. An unshifted mean must move under the same tamper."""
    hist = history()
    sc = statcast_for(hist)

    def unshifted(s):
        built = MS.build(hist, "hitters", s).sort_values(
            ["player_id", "season", "period"])
        g = built.groupby("player_id", sort=False)["sc_xwobacon"]
        return g.transform(
            lambda x: x.ewm(halflife=MS.HALFLIFE, min_periods=1).mean()
        ).to_numpy()

    a = unshifted(sc)
    sc2 = sc.copy()
    sc2.loc[sc2["game_pk"] == str(hist["game_pk"].max()),
            "sc_xwobacon"] = 9.99
    b = unshifted(sc2)
    assert not np.allclose(np.nan_to_num(a, nan=-1.0),
                           np.nan_to_num(b, nan=-1.0)), (
        "an unshifted mean did not move, so the leak test cannot detect a "
        "leak and proves nothing")


def it_is_off_until_a_grade_says_otherwise():
    """Built, joined, and NOT in the spec. The opposing-starter lesson."""
    assert MS.STATCAST_HITTER_FEATURES == [], (
        "Statcast features are in the hitter spec. They must be graded "
        "against the current model before they can reach a board.")
    assert MS.STATCAST_PITCHER_FEATURES == []


SUITES = [
    ("PITCHES BECOME PLAYER-GAMES", [
        ("one row per player per game per side",
         one_row_per_player_per_game_per_side),
        ("counts, not rates, so weekly chunks can be added",
         counts_not_rates_so_chunks_can_be_added),
        ("a barrel is the league's own definition",
         a_barrel_is_the_league_definition),
        ("a changed feed is reported, not guessed at",
         a_changed_feed_is_reported_not_guessed),
        ("an empty fetch is not a crash", an_empty_fetch_is_not_a_crash),
    ]),
    ("PLAYER-GAMES BECOME A FEATURE", [
        ("it reaches the hitter frame", the_feature_reaches_the_frame),
        ("a missing cache degrades instead of failing",
         missing_statcast_is_a_supported_state),
        ("and the join does not fan out", the_join_does_not_fan_out),
    ]),
    ("A LEAK HERE WOULD LOOK LIKE SUCCESS", [
        ("a game cannot see its own Statcast",
         a_game_cannot_see_its_own_statcast),
        ("and that check can actually fail", the_leak_test_can_fail),
        ("the features are OFF until a grade says otherwise",
         it_is_off_until_a_grade_says_otherwise),
    ]),
]


def main() -> int:
    passed = failed = 0
    for title, cases in SUITES:
        print(f"\n{title}")
        print("-" * 66)
        for name, fn in cases:
            try:
                fn()
                print(f"  ok    {name}")
                passed += 1
            except Exception as exc:                           # noqa: BLE001
                print(f"  FAIL  {name}\n        {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
