"""The opposing-starter feature, and the leak it must not have.

Runs as a script: `python test_opp_starter.py`.

The feature is forward-looking on purpose - tonight's probable pitcher is
announced the day before - so the ONLY thing standing between it and a leak is
that the starter's own FORM is measured strictly before the game being
projected. Two of the tests below exist to make that fail if it ever stops
being true, and one of those deliberately breaks it first to prove the check
is capable of firing.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import mlb_sport as MS


def history(days: int = 40, teams=("AAA", "BBB"), seed: int = 0):
    """A small two-team season: one starter each, nine hitters each.

    Dates are DISTINCT. The first version wrote `2026-04-{d % 28 + 1}`, which
    wrapped after 28 days and gave two different games the same date - so
    `to_canonical` derived the same period for both, the rows sorted into an
    order that was not chronological, and the leak test failed against
    perfectly correct code. A fixture that cannot represent time cannot test
    a feature whose whole correctness is about time.
    """
    from datetime import date, timedelta
    start = date(2026, 4, 1)
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        gpk = 900000 + d
        home, away = teams[0], teams[1]
        for team, opp in ((home, away), (away, home)):
            # The starter. AAA's man is an ace, BBB's is not - a difference
            # the model must be able to see.
            ace = team == "AAA"
            rows.append({
                "player_id": f"SP_{team}", "name": f"SP {team}",
                "team": team, "opponent": opp, "date": (start + timedelta(days=d)).isoformat(),
                "game_pk": gpk, "is_pitcher": 1, "position": "SP",
                "innings": 6.0, "batters_faced": 24.0,
                "strikeout": 9.0 if ace else 3.0,
                "hit_allowed": 4.0 if ace else 8.0,
                "walk_allowed": 1.0 if ace else 3.0,
                "earned_run": 2.0 if ace else 4.0,
                "points": 18.0 if ace else 8.0,
            })
            # A long reliever, who must NOT be mistaken for the starter.
            rows.append({
                "player_id": f"RP_{team}", "name": f"RP {team}",
                "team": team, "opponent": opp, "date": (start + timedelta(days=d)).isoformat(),
                "game_pk": gpk, "is_pitcher": 1, "position": "RP",
                "innings": 2.0, "batters_faced": 7.0, "strikeout": 2.0,
                "hit_allowed": 2.0, "walk_allowed": 1.0, "earned_run": 1.0,
                "points": 4.0,
            })
            for i in range(9):
                # Hitters score less against the ace, which is the effect the
                # feature is supposed to carry.
                hard = opp == "AAA"
                rows.append({
                    "player_id": f"H_{team}_{i}", "name": f"H {team} {i}",
                    "team": team, "opponent": opp,
                    "date": (start + timedelta(days=d)).isoformat(),
                    "game_pk": gpk, "is_pitcher": 0,
                    "position": ["C","1B","2B","3B","SS","LF","CF","RF","DH"][i],
                    "plate_appearances": 4.0, "at_bats": 4.0,
                    "single": rng.poisson(0.6 if hard else 1.0),
                    "double": rng.poisson(0.15), "triple": 0.0,
                    "home_run": rng.poisson(0.08 if hard else 0.18),
                    "rbi": rng.poisson(0.3), "run": rng.poisson(0.4),
                    "walk": rng.poisson(0.3), "stolen_base": 0.0,
                    "bat_slot": float(i + 1), "bat_started": 1.0,
                    "points": float(rng.normal(6.0 if hard else 9.0, 3.0)),
                })
    return pd.DataFrame(rows)


def the_starter_is_the_man_who_faced_most_batters():
    """Not the long reliever, who pitched in the same game."""
    q = MS.starter_quality(history(12))
    assert not q.empty
    assert set(q["opp_sp_id"]) == {"SP_AAA", "SP_BBB"}, (
        f"picked {sorted(set(q['opp_sp_id']))} as starters - a reliever has "
        f"been mistaken for one")


def it_joins_to_the_hitter_by_opponent():
    """A hitter's opp_sp is the pitcher on the OTHER team, not his own."""
    q = MS.starter_quality(history(12))
    # AAA's starter is keyed to team AAA; a BBB hitter has opponent AAA.
    aaa = q[q["team"] == "AAA"]
    assert set(aaa["opp_sp_id"]) == {"SP_AAA"}


def the_ace_and_the_rest_are_distinguishable():
    """If both starters look the same the feature carries nothing."""
    q = MS.starter_quality(history(30)).dropna(subset=["opp_sp_k_rate"])
    by = q.groupby("opp_sp_id")["opp_sp_k_rate"].mean()
    assert by["SP_AAA"] > by["SP_BBB"] * 1.5, (
        f"the ace's strikeout rate {by['SP_AAA']:.3f} is not clearly above "
        f"the other starter's {by['SP_BBB']:.3f}")


def form_cannot_see_its_own_game():
    """The leak test.

    A starter's form on the row describing game N must be built only from
    games before N. Blowing up a single start and checking that the form
    attached to THAT start does not move is the direct test.
    """
    hist = history(30)
    q = MS.starter_quality(hist).sort_values(["opp_sp_id", "game_pk"])
    before = q[q["opp_sp_id"] == "SP_AAA"]["opp_sp_k_rate"].to_numpy()

    tampered = hist.copy()
    # Ruin the LAST AAA start: no strikeouts at all.
    mask = (tampered["player_id"] == "SP_AAA")
    last = tampered[mask]["game_pk"].max()
    tampered.loc[mask & (tampered["game_pk"] == last), "strikeout"] = 0.0
    after = MS.starter_quality(tampered).sort_values(
        ["opp_sp_id", "game_pk"])
    after = after[after["opp_sp_id"] == "SP_AAA"]["opp_sp_k_rate"].to_numpy()

    a = np.nan_to_num(before, nan=-1.0)
    b = np.nan_to_num(after, nan=-1.0)
    assert np.allclose(a, b), (
        "changing a start changed the form attached to that same start - the "
        "feature is looking at the game it is describing")


def the_leak_test_can_fail():
    """Prove the check above is capable of firing.

    An unshifted mean DOES see its own game, so the same tamper must move it.
    Without this, `form_cannot_see_its_own_game` passing proves nothing.
    """
    hist = history(30)

    def unshifted(h):
        s = MS._starter_rows(h)
        g = s.groupby("player_id", sort=False)
        s["k"] = g["_k"].transform(
            lambda x: x.ewm(halflife=MS.SP_HALFLIFE, min_periods=1).mean())
        s = s[s["player_id"] == "SP_AAA"].sort_values("game_pk")
        return s["k"].to_numpy()

    before = unshifted(hist)
    tampered = hist.copy()
    mask = tampered["player_id"] == "SP_AAA"
    last = tampered[mask]["game_pk"].max()
    tampered.loc[mask & (tampered["game_pk"] == last), "strikeout"] = 0.0
    after = unshifted(tampered)
    assert not np.allclose(before, after), (
        "an unshifted mean did not move when its own game changed, so the "
        "leak test cannot detect a leak")


def the_feature_reaches_the_hitter_rows():
    hist = history(40)
    built = MS.build(hist, "hitters")
    for c in ("opp_sp_k_rate", "opp_sp_baserunners"):
        assert c in built.columns, f"{c} never reached the hitter frame"
    have = float(built["opp_sp_k_rate"].notna().mean())
    assert have > 0.8, f"only {100 * have:.0f}% of hitter rows know the starter"


def the_join_does_not_duplicate_rows():
    """The fan-out guard. One hitter row, one opposing starter."""
    hist = history(30)
    hitters, _ = MS.split(hist)
    built = MS.build(hist, "hitters")
    assert len(built) == len(hitters), (
        f"{len(hitters)} hitter rows became {len(built)} after the join")


def current_form_uses_the_latest_start():
    """A board needs his most recent start included, not shifted away."""
    hist = history(30)
    now = MS.current_starter_form(hist)
    assert set(now["opp_sp_id"]) == {"SP_AAA", "SP_BBB"}
    assert now["opp_sp_k_rate"].notna().all()
    hist2 = hist.copy()
    mask = hist2["player_id"] == "SP_AAA"
    last = hist2[mask]["game_pk"].max()
    hist2.loc[mask & (hist2["game_pk"] == last), "strikeout"] = 0.0
    now2 = MS.current_starter_form(hist2)
    a = float(now[now["opp_sp_id"] == "SP_AAA"]["opp_sp_k_rate"].iloc[0])
    b = float(now2[now2["opp_sp_id"] == "SP_AAA"]["opp_sp_k_rate"].iloc[0])
    assert b < a, ("the latest start did not affect tonight's form, so it is "
                   "being shifted away when it should not be")


SUITES = [
    ("WHO IS THE STARTER", [
        ("the man who faced the most batters, not the long reliever",
         the_starter_is_the_man_who_faced_most_batters),
        ("keyed to the team he threw FOR, so a hitter finds him by opponent",
         it_joins_to_the_hitter_by_opponent),
        ("an ace and a journeyman are clearly different",
         the_ace_and_the_rest_are_distinguishable),
    ]),
    ("THE FEATURE IS FORWARD-LOOKING, HIS FORM IS NOT", [
        ("a start cannot see itself", form_cannot_see_its_own_game),
        ("and that check can actually fail", the_leak_test_can_fail),
        ("tonight's form DOES include his latest start",
         current_form_uses_the_latest_start),
    ]),
    ("IT REACHES THE MODEL WITHOUT BREAKING THE FRAME", [
        ("every hitter row carries the opposing starter",
         the_feature_reaches_the_hitter_rows),
        ("and the join does not fan out", the_join_does_not_duplicate_rows),
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
    logging = __import__("logging")
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
