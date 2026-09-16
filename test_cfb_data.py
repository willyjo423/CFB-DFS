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

def test_middle_initial_is_bridged_when_unambiguous():
    """Tyler J. Williams on the board, Tyler Williams in CFBD.

    No exact reduction bridges this - deleting a middle token is not a
    spelling difference - so it needs its own pass.
    """
    board = pd.DataFrame([{"name": "Tyler J. Williams", "salary": 3000,
                           "position": "WR", "team": "UGA"}])
    board["keys"] = board["name"].map(D.name_keys)
    hist = pd.DataFrame([{"athlete_id": "9", "name": "Tyler Williams"}])
    hist["keys"] = hist["name"].map(D.name_keys)
    assert D.attach_history(board, hist)["athlete_id"].iloc[0] == "9"


def test_middle_initial_is_NOT_bridged_when_ambiguous():
    """Two Tyler Williamses. A coin flip on a real salary beats no miss.

    This is the test that makes the pass above safe. Without the uniqueness
    requirement it would attach one of these two at random, and a wrong
    projection is worse than an absent one - the absent one is dropped from
    the pool, the wrong one gets rostered.
    """
    board = pd.DataFrame([{"name": "Tyler J. Williams", "salary": 3000,
                           "position": "WR", "team": "UGA"}])
    board["keys"] = board["name"].map(D.name_keys)
    hist = pd.DataFrame([{"athlete_id": "9", "name": "Tyler Williams"},
                         {"athlete_id": "10", "name": "Tyler Adam Williams"}])
    hist["keys"] = hist["name"].map(D.name_keys)
    got = D.attach_history(board, hist)["athlete_id"].iloc[0]
    assert got is None or pd.isna(got), got


def test_exact_match_still_wins_over_the_reduced_pass():
    """A real middle name must not be discarded when it identifies someone."""
    board = pd.DataFrame([{"name": "Tyler Adam Williams", "salary": 3000,
                           "position": "WR", "team": "UGA"}])
    board["keys"] = board["name"].map(D.name_keys)
    hist = pd.DataFrame([{"athlete_id": "9", "name": "Tyler Adam Williams"},
                         {"athlete_id": "10", "name": "Tyler Williams"}])
    hist["keys"] = hist["name"].map(D.name_keys)
    assert D.attach_history(board, hist)["athlete_id"].iloc[0] == "9"


def test_join_matches_across_spellings():
    board = pd.DataFrame([{"name": "AJ Swann", "salary": 8500,
                           "position": "QB", "team": "ARK"}])
    board["keys"] = board["name"].map(D.name_keys)
    hist = pd.DataFrame([{"athlete_id": "77", "name": "A.J. Swann"}])
    hist["keys"] = hist["name"].map(D.name_keys)
    out = D.attach_history(board, hist)
    assert out["athlete_id"].iloc[0] == "77", out


def _board(rows):
    return pd.DataFrame([
        {"name": n, "salary": s, "position": p, "team": "X",
         "athlete_id": aid, "dk_points_per_game": ppg}
        for n, s, p, aid, ppg in rows])


def test_a_miss_with_production_is_not_the_same_as_a_miss_without():
    """Price cannot tell these apart. Published production can.

    Both boards miss exactly one player of four, and the misses are priced
    identically at $4,500. One has never taken a snap - DraftKings scores him
    0.0 and prices him at $4,500 precisely BECAUSE it has no data on him.
    The other is producing 15.2 a game and we lost him.

    A salary-based test calls these the same thing, which is why the first
    version of this check passed a board that was losing real players.
    """
    harmless = _board([
        ("a", 9000, "QB", "1", 24.0),
        ("b", 7000, "WR", "2", 11.0),
        ("c", 6000, "RB", "3", 9.0),
        ("backup", 4500, "QB", None, 0.0),
    ])
    assert D.real_misses(harmless).empty
    assert "nothing was lost" in D.join_quality(harmless)

    real = _board([
        ("a", 9000, "QB", "1", 24.0),
        ("b", 7000, "WR", "2", 11.0),
        ("c", 6000, "RB", "3", 9.0),
        ("producer", 4500, "WR", None, 15.2),
    ])
    bad = D.real_misses(real)
    assert list(bad["name"]) == ["producer"], list(bad["name"])
    assert "real misses" in D.join_quality(real)


def test_the_gate_is_not_fooled_by_a_board_full_of_backups():
    """The failure mode that made the first gate useless.

    Sixty backups with no snaps and four producers, all matched. A raw match
    rate reads 6% and fails. The gate that matters reads zero real misses and
    passes, because nothing was lost.
    """
    rows = [(f"backup{i}", 3000, "WR", None, 0.0) for i in range(60)]
    rows += [(f"star{i}", 9000, "QB", str(i), 20.0) for i in range(4)]
    board = _board(rows)
    assert board["athlete_id"].notna().mean() < 0.10
    assert D.real_misses(board).empty


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


def _slate_20260919():
    """The real Saturday slate the first live verify run mapped 6 of 24 on.

    Twelve fixtures, twenty-four codes, and CFBD team records carrying the
    abbreviations that endpoint actually publishes. This is the regression
    test for the whole team-map rewrite.
    """
    fixtures = [
        ("UTST", "UTAH", "Utah State", "Utah"),
        ("UGA", "ARK", "Georgia", "Arkansas"),
        ("KENT", "OSU", "Kent State", "Ohio State"),
        ("MSST", "SCAR", "Mississippi State", "South Carolina"),
        ("UK", "TA&M", "Kentucky", "Texas A&M"),
        ("FSU", "BAMA", "Florida State", "Alabama"),
        ("BUFF", "PSU", "Buffalo", "Penn State"),
        ("STAN", "DUKE", "Stanford", "Duke"),
        ("RU", "USC", "Rutgers", "USC"),
        ("ASU", "CLEM", "Arizona State", "Clemson"),
        ("KU", "UL", "Kansas", "Louisiana"),
        ("SMU", "UNC", "SMU", "North Carolina"),
    ]
    board = pd.DataFrame([{"game": f"{a} @ {h}", "team": a, "opponent": h,
                           "is_home": 0} for a, h, _, _ in fixtures])
    games = [{"away_team": aw, "home_team": hm} for _, _, aw, hm in fixtures]
    abbrev = {
        "Utah State": "USU", "Utah": "UTAH", "Georgia": "UGA",
        "Arkansas": "ARK", "Kent State": "KENT", "Ohio State": "OSU",
        "Mississippi State": "MSST", "South Carolina": "SCAR",
        "Kentucky": "UK", "Texas A&M": "TAMU", "Florida State": "FSU",
        "Alabama": "ALA", "Buffalo": "BUFF", "Penn State": "PSU",
        "Stanford": "STAN", "Duke": "DUKE", "Rutgers": "RUTG", "USC": "USC",
        "Arizona State": "ASU", "Clemson": "CLEM", "Kansas": "KU",
        "Louisiana": "UL", "SMU": "SMU", "North Carolina": "UNC",
    }
    teams = [{"school": s, "abbreviation": a} for s, a in abbrev.items()]
    expected = {}
    for away_code, home_code, away, home in fixtures:
        expected[away_code] = away
        expected[home_code] = home
    return board, games, teams, expected


def test_real_slate_maps_every_code():
    board, games, teams, expected = _slate_20260919()
    m = D.fixture_team_map(board, games, teams)
    missing = {c for c in expected if c not in m}
    wrong = {c: (m[c], expected[c]) for c in expected
             if c in m and m[c] != expected[c]}
    assert not wrong, f"WRONG mappings: {wrong}"
    assert not missing, f"unmapped: {sorted(missing)}"


def test_uk_and_ku_do_not_deadlock():
    """The collision the two-tier alias scheme exists to prevent.

    Constructing "initials + U" makes Kentucky claim KU, and "U + first
    letter" makes Kansas claim UK. With one flat tier each code proposes both
    schools, both fixtures see two candidates, and neither resolves.

    Getting this to fail took three attempts, and the first two are the
    lesson. On the full slate the collision is survivable, because other
    fixtures claim schools and prune these candidate lists by elimination. It
    is still survivable with only these two fixtures, as long as ONE home code
    is officially claimed - that fixture resolves and rescues the other.

    The deadlock needs both home codes to be underivable, which is precisely
    the BAMA and TA&M situation on the real board: Alabama's abbreviation is
    ALA and Texas A&M's is TAMU, so neither DraftKings code is claimed by
    anyone. Then nothing can be resolved first and single-tier maps zero of
    four. Two earlier versions of this test passed against the broken
    implementation, which is the same as not testing it.
    """
    board = pd.DataFrame([
        {"game": "UK @ BAMA", "team": "UK", "opponent": "BAMA", "is_home": 0},
        {"game": "KU @ TA&M", "team": "KU", "opponent": "TA&M", "is_home": 0},
    ])
    games = [{"away_team": "Kentucky", "home_team": "Alabama"},
             {"away_team": "Kansas", "home_team": "Texas A&M"}]
    teams = [{"school": "Kentucky", "abbreviation": "UK"},
             {"school": "Kansas", "abbreviation": "KU"},
             {"school": "Alabama", "abbreviation": "ALA"},
             {"school": "Texas A&M", "abbreviation": "TAMU"}]
    m = D.fixture_team_map(board, games, teams)
    assert m.get("UK") == "Kentucky", m
    assert m.get("KU") == "Kansas", m
    assert m.get("BAMA") == "Alabama", m
    assert m.get("TA&M") == "Texas A&M", m


def test_codes_no_rule_can_derive_still_resolve():
    """BAMA, TA&M and UTST are derivable from nothing. Fixtures carry them.

    Alabama's abbreviation is ALA, Texas A&M's is TAMU, Utah State's is USU.
    None of the three DraftKings codes is a prefix, an initialism or a
    substring of its school. Each is solved only by who its opponent is.
    """
    board, games, teams, _ = _slate_20260919()
    m = D.fixture_team_map(board, games, teams)
    assert m.get("BAMA") == "Alabama", m.get("BAMA")
    assert m.get("TA&M") == "Texas A&M", m.get("TA&M")
    assert m.get("UTST") == "Utah State", m.get("UTST")


def test_without_teams_payload_it_says_so_and_still_tries():
    """The fallback must degrade, not crash - and must not claim success."""
    board, games, _, _ = _slate_20260919()
    m = D.fixture_team_map(board, games, teams=None)
    assert isinstance(m, dict)
    assert len(m) < 24, "name matching alone should NOT solve every code"


def test_confidently_wrong_proposals_are_discarded():
    """UTST, exactly as the live run hit it.

    "UT Martin" builds UTST from its first token and Utah Tech from its
    initials; Utah State builds USST and UTAHST and never UTST. So UTST
    proposed four schools with total confidence and every one was wrong,
    which is strictly worse than proposing nothing - a code proposing nothing
    is treated as free and solved by its opponent, and this one was not.

    The recovery is to notice the fixture has ZERO candidates, which cannot
    mean ambiguity, and throw the proposals away.
    """
    board = pd.DataFrame([{"game": "UTST @ UTAH", "team": "UTST",
                           "opponent": "UTAH", "is_home": 0}])
    games = [{"away_team": "Utah State", "home_team": "Utah"},
             {"away_team": "UT Martin", "home_team": "Memphis"},
             {"away_team": "Utah Tech", "home_team": "Idaho"}]
    teams = [{"school": s, "abbreviation": a} for s, a in [
        ("Utah State", "USU"), ("Utah", "UTAH"), ("UT Martin", "UTM"),
        ("Memphis", "MEM"), ("Utah Tech", "UTU"), ("Idaho", "IDHO")]]
    m = D.fixture_team_map(board, games, teams)
    assert m.get("UTAH") == "Utah", m
    assert m.get("UTST") == "Utah State", m


def test_neutral_site_home_away_disagreement():
    """SMU and Louisiana, exactly as the live run left them.

    Both codes proposed exactly one school, both correct, across a 596-game
    window - and no fixture contained them in the orientation DraftKings
    printed. At a neutral site "home" is a bookkeeping choice and the two
    sources are free to make it differently.
    """
    board = pd.DataFrame([
        {"game": "SMU @ UL", "team": "SMU", "opponent": "UL", "is_home": 0},
        {"game": "BUFF @ PSU", "team": "BUFF", "opponent": "PSU",
         "is_home": 0},
    ])
    # CFBD has Louisiana as the visitor; DraftKings printed it as the host.
    games = [{"away_team": "Louisiana", "home_team": "SMU"},
             {"away_team": "Buffalo", "home_team": "Penn State"}]
    teams = [{"school": s, "abbreviation": a} for s, a in [
        ("Louisiana", "UL"), ("SMU", "SMU"), ("Buffalo", "BUFF"),
        ("Penn State", "PSU")]]
    m = D.fixture_team_map(board, games, teams)
    assert m.get("SMU") == "SMU", m
    assert m.get("UL") == "Louisiana", m
    # The normally-oriented fixture must still map correctly.
    assert m.get("BUFF") == "Buffalo", m
    assert m.get("PSU") == "Penn State", m


def test_swap_does_not_override_a_correct_forward_match():
    """The reversed pass must never win where the strict one could.

    Both fixtures are correctly oriented and involve the same four schools in
    different pairings. A swap tried too eagerly pairs the wrong teams, which
    hands players the wrong opponent and the wrong implied total - confident
    numbers that are wrong in every row.
    """
    board = pd.DataFrame([
        {"game": "MICH @ OSU", "team": "MICH", "opponent": "OSU",
         "is_home": 0},
        {"game": "MINN @ ORE", "team": "MINN", "opponent": "ORE",
         "is_home": 0},
    ])
    games = [{"away_team": "Michigan", "home_team": "Ohio State"},
             {"away_team": "Minnesota", "home_team": "Oregon"}]
    teams = [{"school": s, "abbreviation": a} for s, a in [
        ("Michigan", "MICH"), ("Ohio State", "OSU"),
        ("Minnesota", "MINN"), ("Oregon", "ORE")]]
    m = D.fixture_team_map(board, games, teams)
    assert m == {"MICH": "Michigan", "OSU": "Ohio State",
                 "MINN": "Minnesota", "ORE": "Oregon"}, m


def test_relaxation_does_not_invent_a_mapping():
    """The relaxation must not turn 'no answer' into 'any answer'.

    Two fixtures, both codes underivable, two plausible games. Zero
    candidates becomes several candidates, not one, and several is still
    unsolved. A relaxation that guessed here would be worse than the bug it
    fixes, because an unsolved code is recoverable and a wrong one is not.
    """
    board = pd.DataFrame([
        {"game": "QQ @ ZZ", "team": "QQ", "opponent": "ZZ", "is_home": 0},
        {"game": "XX @ YY", "team": "XX", "opponent": "YY", "is_home": 0},
    ])
    games = [{"away_team": "Michigan", "home_team": "Ohio State"},
             {"away_team": "Minnesota", "home_team": "Oregon"}]
    teams = [{"school": s, "abbreviation": a} for s, a in [
        ("Michigan", "MICH"), ("Ohio State", "OSU"),
        ("Minnesota", "MINN"), ("Oregon", "ORE")]]
    m = D.fixture_team_map(board, games, teams)
    assert not m, f"should have solved nothing, got {m}"


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
