"""Tests for the baseball-specific half: the doubleheader packing.

There is exactly one thing here that the engine's own tests cannot cover,
because no other sport in this project has it: two games on one date.

Every other sport gets to treat a period and a game as the same object. When
baseball did too, the frame validator caught the duplicate keys - but the
duplicate keys were the *cheap* symptom. The expensive one was silent: every
`share_*` feature is computed by grouping on (team, season, period), so on a
doubleheader date a hitter's share of his team's plate appearances would have
been measured against two games' worth of team totals and come out at half
its true value. No error, no warning, just a feature quietly reporting the
wrong number on precisely the days a hitter has the most opportunity.

So the share test below is the one that matters. The rest are guard rails
around it.

Run: python test_mlb_sport.py
"""

from __future__ import annotations

import pandas as pd

import mlb_sport as MS
from engine import features as EF


def _rows(recs: list[dict]) -> pd.DataFrame:
    """A minimal history frame: only what to_canonical and build need."""
    base = {
        "plate_appearances": 0.0, "at_bats": 0.0, "single": 0.0,
        "double": 0.0, "triple": 0.0, "home_run": 0.0, "rbi": 0.0,
        "run": 0.0, "walk": 0.0, "stolen_base": 0.0,
        "position": "SS", "is_pitcher": 0, "points": 0.0, "is_home": 1,
        "opponent": "OPP", "name": "somebody",
    }
    return pd.DataFrame([{**base, **r} for r in recs])


def _doubleheader() -> pd.DataFrame:
    """One team, one date, two games. Two hitters in both games.

    In EACH game the team takes 10 plate appearances and our man takes 5, so
    his share is 0.5 in each. Pooled across the pair it would read 0.25.
    """
    recs = []
    for gpk, pa_ours in (("700001", 5.0), ("700002", 5.0)):
        recs.append({"player_id": "aaa", "team": "BOS", "date": "2026-07-04",
                     "game_pk": gpk, "plate_appearances": pa_ours,
                     "at_bats": pa_ours, "single": 1.0, "points": 3.0,
                     "name": "our man"})
        recs.append({"player_id": "bbb", "team": "BOS", "date": "2026-07-04",
                     "game_pk": gpk, "plate_appearances": 5.0,
                     "at_bats": 5.0, "single": 2.0, "points": 6.0,
                     "name": "team-mate"})
    return _rows(recs)


def test_a_doubleheader_gets_two_periods():
    out = MS.to_canonical(_doubleheader())
    ours = out[out["player_id"] == "aaa"]
    assert len(ours) == 2, "the fixture should have two games"
    assert ours["period"].nunique() == 2, (
        f"both games landed on period {ours['period'].tolist()} - a period "
        f"must identify a game, not a date")


def test_no_duplicate_player_period_keys():
    out = MS.to_canonical(_doubleheader())
    dupes = int(out.duplicated(["player_id", "season", "period"]).sum())
    assert dupes == 0, f"{dupes} duplicate keys survived the packing"


def test_a_single_game_day_uses_the_first_slot():
    one = _rows([{"player_id": "aaa", "team": "BOS", "date": "2026-07-04",
                  "game_pk": "700001", "plate_appearances": 4.0,
                  "points": 3.0}])
    out = MS.to_canonical(one)
    doy = pd.Timestamp("2026-07-04").dayofyear
    assert int(out["period"].iloc[0]) == doy * MS.SLOTS, (
        "an ordinary day must sit at slot 0, or every existing day index "
        "shifts underneath the grader")


def test_a_day_and_its_periods_round_trip():
    for day in (1, 182, 258, 366):
        first, last = MS.periods_of(day)
        assert MS.day_of(first) == day
        assert MS.day_of(last) == day
        assert last - first == MS.SLOTS - 1


def test_periods_stay_in_calendar_order():
    """Packing must not let one date overtake the next."""
    two_days = _rows([
        {"player_id": "aaa", "team": "BOS", "date": "2026-07-04",
         "game_pk": "700001", "points": 1.0},
        {"player_id": "aaa", "team": "BOS", "date": "2026-07-04",
         "game_pk": "700002", "points": 2.0},
        {"player_id": "aaa", "team": "BOS", "date": "2026-07-05",
         "game_pk": "700003", "points": 3.0},
    ])
    out = MS.to_canonical(two_days).sort_values("period")
    assert out["date"].tolist() == ["2026-07-04", "2026-07-04", "2026-07-05"]


def test_a_share_is_measured_within_one_game():
    """The reason all of this exists."""
    built = EF.build(MS.to_canonical(_doubleheader()), MS.HITTERS,
                     validate=False)
    ours = built[built["player_id"] == "aaa"]
    shares = ours["share_plate_appearances"].tolist()
    assert all(abs(s - 0.5) < 1e-9 for s in shares), (
        f"share_plate_appearances came out {shares}, not [0.5, 0.5]. "
        f"The two games were pooled into one team-period.")


def test_the_game_order_follows_the_game_id():
    """Slot 0 is the earlier game id, whatever order the rows arrive in."""
    df = _doubleheader().iloc[::-1].reset_index(drop=True)
    out = MS.to_canonical(df)
    ours = out[out["player_id"] == "aaa"].sort_values("period")
    assert ours["game_pk"].tolist() == ["700001", "700002"]


def test_a_missing_game_id_does_not_crash_or_invent_a_period():
    df = _rows([{"player_id": "aaa", "team": "BOS", "date": "2026-07-04",
                 "game_pk": None, "points": 1.0}])
    out = MS.to_canonical(df)
    doy = pd.Timestamp("2026-07-04").dayofyear
    assert int(out["period"].iloc[0]) == doy * MS.SLOTS


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
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(main())
