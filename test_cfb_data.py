"""Tests that can fail.

A test that passes no matter what the code does is worse than no test, because
it buys confidence without paying for it. Every case here was chosen because a
plausible implementation gets it wrong, and several of them correspond to bugs
this project has already shipped once.

pytest is not installable in the container these were written in, so each test
is a plain function and `main()` runs them. That is deliberate: the tests must
be runnable by the same command everywhere rather than needing a runner that
may or may not be there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cfb_data as D


# ------------------------------------------------------------------- naming

def test_initials_and_periods_both_match():
    """AJ Swann cost this project a $8,500 quarterback once. Not twice.

    DraftKings writes "AJ Swann", CFBD writes "A.J. Swann". Period-to-space
    gives "a j swann"; period-to-nothing gives "aj swann". Either rule alone
    misses. Both together must overlap.
    """
    dk = D.name_keys("AJ Swann")
    cfbd = D.name_keys("A.J. Swann")
    assert dk & cfbd, f"no shared key: {dk} vs {cfbd}"


def test_no_space_period_still_matches():
    """The opposite failure: St.Brown vs St. Brown."""
    assert D.name_keys("Amon-Ra St.Brown") & D.name_keys("Amon-Ra St. Brown")


def test_suffixes_and_ordering():
    assert D.name_keys("Smith, Cam Jr.") & D.name_keys("Cam Smith")


def test_different_players_do_not_collide():
    """The reason matching is exact rather than fuzzy.

    These two names are one edit apart. An edit-distance matcher pairs them;
    on a college roster of 120 with brothers and cousins, that is a wrong
    projection attached to a real salary, which is worse than a miss.
    """
    assert not (D.name_keys("Jaden Rashada") & D.name_keys("Jalen Rashada"))


# ------------------------------------------------------------------ scoring

def test_full_ppr_and_bonus_step():
    """College is full PPR, and the yardage bonus is a step not a slope."""
    df = pd.DataFrame([
        # 99 receiving yards, 10 catches, no bonus
        {"rec": 10, "rec_yards": 99, "rec_td": 0},
        # 100 receiving yards, 10 catches, +3 bonus
        {"rec": 10, "rec_yards": 100, "rec_td": 0},
    ])
    for f in D.STAT_FIELDS:
        if f not in df:
            df[f] = 0.0
    pts = D.fantasy_points(df)
    assert abs(pts.iloc[0] - (10 + 9.9)) < 1e-6, pts.iloc[0]
    assert abs(pts.iloc[1] - (10 + 10.0 + 3.0)) < 1e-6, pts.iloc[1]
    # One yard of talent, three points of scoring. This is the whole argument
    # for simulating a distribution instead of projecting a mean.
    assert pts.iloc[1] - pts.iloc[0] > 3.0


def test_interception_and_fumble_are_negative():
    df = pd.DataFrame([{f: 0.0 for f in D.STAT_FIELDS}])
    df.loc[0, "interception"] = 2
    df.loc[0, "fumble_lost"] = 1
    assert D.fantasy_points(df).iloc[0] == -3.0


def test_completions_parse_from_slash_format():
    """C/ATT arrives as "18/25". float() rejects it and NaN deletes the row."""
    assert D._number("18/25") == 18.0
    assert D._number("1,234") == 1234.0
    assert np.isnan(D._number(None))


# ----------------------------------------------------------------- flatten

def _payload():
    return [{
        "id": 401,
        "teams": [
            {"team": "Alabama", "homeAway": "home", "categories": [
                {"name": "passing", "types": [
                    {"name": "YDS", "athletes": [{"id": 1, "name": "Ty Simpson",
                                                  "stat": "305"}]},
                    {"name": "TD", "athletes": [{"id": 1, "name": "Ty Simpson",
                                                 "stat": "3"}]},
                ]},
                {"name": "receiving", "types": [
                    {"name": "REC", "athletes": [{"id": 2, "name": "R Williams",
                                                  "stat": "7"}]},
                    {"name": "YDS", "athletes": [{"id": 2, "name": "R Williams",
                                                  "stat": "112"}]},
                ]},
            ]},
            {"team": "Georgia", "homeAway": "away", "categories": [
                {"name": "rushing", "types": [
                    {"name": "YDS", "athletes": [{"id": 3, "name": "N Frazier",
                                                  "stat": "64"}]},
                ]},
            ]},
        ],
    }]


def test_flatten_shapes_and_opponents():
    wide = D.flatten_player_games(_payload())
    assert len(wide) == 3, wide
    ty = wide[wide["athlete_id"] == "1"].iloc[0]
    assert ty["pass_yards"] == 305 and ty["pass_td"] == 3
    assert ty["school"] == "Alabama" and ty["opponent"] == "Georgia"
    assert ty["is_home"] == 1
    geo = wide[wide["athlete_id"] == "3"].iloc[0]
    assert geo["opponent"] == "Alabama" and geo["is_home"] == 0
    # Every scoring field must exist even when nobody recorded it, or the
    # scorer silently skips it.
    for f in D.STAT_FIELDS:
        assert f in wide, f


def test_flatten_scores_correctly():
    wide = D.flatten_player_games(_payload())
    wide["points"] = D.fantasy_points(wide)
    ty = wide[wide["athlete_id"] == "1"].iloc[0]
    # 305 * 0.04 + 3 * 4 = 12.2 + 12 = 24.2, plus the 300-yard bonus
    assert abs(ty["points"] - (12.2 + 12.0 + 3.0)) < 1e-6, ty["points"]
    rec = wide[wide["athlete_id"] == "2"].iloc[0]
    # 7 catches + 11.2 yards + 100-yard bonus
    assert abs(rec["points"] - (7 + 11.2 + 3.0)) < 1e-6, rec["points"]


# ----------------------------------------------------------------- captains

def _showdown_board():
    """Six players, each listed twice: face value and 1.5x."""
    rows = []
    for name, base in [("A Player", 10000), ("B Player", 8000),
                       ("C Player", 6000), ("D Player", 4000)]:
        rows.append({"name": name, "salary": base, "position": "WR"})
        rows.append({"name": name, "salary": base * 1.5, "position": "WR"})
    return pd.DataFrame(rows)


def test_captain_rows_are_found():
    out = D._mark_captains(_showdown_board(), 1.5)
    assert out["is_captain"].sum() == 4, out
    # Every captain row must be the expensive one of its pair.
    for name, g in out.groupby("name"):
        cap = g[g["is_captain"] == 1]["salary"].iloc[0]
        flex = g[g["is_captain"] == 0]["salary"].iloc[0]
        assert cap > flex, (name, cap, flex)


def test_classic_board_has_no_captains():
    """The test that can fail: a detector that flags everything is useless.

    On a Classic board each player appears once, at one price. Nothing should
    be marked, even though prices vary enormously across players.
    """
    df = pd.DataFrame([{"name": f"P{i}", "salary": s, "position": "WR"}
                       for i, s in enumerate([3000, 4500, 7000, 11000])])
    out = D._mark_captains(df, 1.5)
    assert out["is_captain"].sum() == 0, out


# --------------------------------------------------------------------- join

def test_join_matches_across_spellings():
    board = pd.DataFrame([{"name": "AJ Swann", "salary": 8500,
                           "position": "QB", "team": "ARK"}])
    board["keys"] = board["name"].map(D.name_keys)
    hist = pd.DataFrame([{"athlete_id": "77", "name": "A.J. Swann"}])
    hist["keys"] = hist["name"].map(D.name_keys)
    out = D.attach_history(board, hist)
    assert out["athlete_id"].iloc[0] == "77", out


def test_join_quality_distinguishes_cheap_from_expensive_misses():
    """Counting misses cannot tell a $3,000 miss from an $8,500 one.

    Both boards below miss exactly one player of four. One miss is harmless
    and one is a disaster, and the report must say which is which.
    """
    cheap = pd.DataFrame([
        {"name": "a", "salary": 9000, "position": "QB", "team": "X",
         "athlete_id": "1"},
        {"name": "b", "salary": 7000, "position": "WR", "team": "X",
         "athlete_id": "2"},
        {"name": "c", "salary": 6000, "position": "RB", "team": "X",
         "athlete_id": "3"},
        {"name": "d", "salary": 3000, "position": "TE", "team": "X",
         "athlete_id": None},
    ])
    text = D.join_quality(cheap)
    assert "allowed to fail" in text, text

    dear = cheap.copy()
    dear.loc[0, "athlete_id"] = None      # the $9,000 quarterback
    dear.loc[3, "athlete_id"] = "4"
    text2 = D.join_quality(dear)
    assert "these matter" in text2, text2
    assert "9,000" in text2, text2


# ----------------------------------------------------------------- fixtures

def test_ul_is_resolved_by_who_it_plays_not_by_its_name():
    """The case that cannot be solved by string comparison.

    `UL` is Louisiana to DraftKings. It reads equally well as Louisville, and
    both are real teams playing that week. Only the opponent settles it - and
    the map must get it right for the reason, not by luck of alphabetical
    ordering.
    """
    board = pd.DataFrame([
        {"game": "UL @ TXST", "team": "UL", "opponent": "TXST", "is_home": 0},
        {"game": "LOU @ MIA", "team": "LOU", "opponent": "MIA", "is_home": 0},
    ])
    games = [
        {"away_team": "Louisiana", "home_team": "Texas State"},
        {"away_team": "Louisville", "home_team": "Miami"},
    ]
    m = D.fixture_team_map(board, games)
    assert m["UL"] == "Louisiana", m
    assert m["LOU"] == "Louisville", m
    assert m["TXST"] == "Texas State", m


def test_ambiguous_fixture_is_reported_not_guessed():
    """Two CFBD games matching one DraftKings fixture must resolve to neither.

    Guessing here produces a full board of confident numbers carrying the
    wrong opponent, the wrong implied total and the wrong correlation group.
    An unsolved code is recoverable; a silently wrong one is not.
    """
    board = pd.DataFrame([{"game": "M @ O", "team": "M", "opponent": "O",
                           "is_home": 0}])
    games = [{"away_team": "Michigan", "home_team": "Ohio State"},
             {"away_team": "Minnesota", "home_team": "Oregon"}]
    m = D.fixture_team_map(board, games)
    assert "M" not in m, m


def test_solving_one_fixture_unlocks_another():
    """The fixpoint loop's actual claim, which nothing else here tests.

    ZZ @ YY is unsolvable on its own: neither code resembles any school, so
    both CFBD games are candidates. TENN @ FLA is solvable immediately, and
    claiming Tennessee and Florida removes that game from ZZ @ YY's candidate
    list, leaving exactly one.

    A single-pass implementation leaves ZZ and YY unmapped and passes every
    other test in this file. That is the point of writing this one.
    """
    board = pd.DataFrame([
        {"game": "ZZ @ YY", "team": "ZZ", "opponent": "YY", "is_home": 0},
        {"game": "TENN @ FLA", "team": "TENN", "opponent": "FLA",
         "is_home": 0},
    ])
    games = [{"away_team": "Alabama", "home_team": "Georgia"},
             {"away_team": "Tennessee", "home_team": "Florida"}]
    m = D.fixture_team_map(board, games)
    assert m.get("TENN") == "Tennessee", m
    assert m.get("ZZ") == "Alabama", m
    assert m.get("YY") == "Georgia", m


def test_two_codes_never_claim_the_same_school():
    """Double-assignment would put two DraftKings teams in one CFBD game."""
    board = pd.DataFrame([
        {"game": "AA @ BB", "team": "AA", "opponent": "BB", "is_home": 0},
        {"game": "CC @ DD", "team": "CC", "opponent": "DD", "is_home": 0},
    ])
    games = [{"away_team": "Alabama", "home_team": "Georgia"},
             {"away_team": "Tennessee", "home_team": "Florida"}]
    m = D.fixture_team_map(board, games)
    assert len(set(m.values())) == len(m), m


def test_abbreviations_that_must_work():
    board = pd.DataFrame([
        {"game": "TA&M @ BAMA", "team": "TA&M", "opponent": "BAMA",
         "is_home": 0},
    ])
    games = [{"away_team": "Texas A&M", "home_team": "Alabama"},
             {"away_team": "Rutgers", "home_team": "Nebraska"}]
    m = D.fixture_team_map(board, games)
    assert m.get("BAMA") == "Alabama", m
    assert m.get("TA&M") == "Texas A&M", m


# --------------------------------------------------------------------- run

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
    if failed:
        print("\nFAILURES:")
        for name, exc in failed:
            print(f"  {name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
