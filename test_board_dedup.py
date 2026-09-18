"""One row per player on the board, and the Max Muncy problem.

Runs as a script: `python test_board_dedup.py`.

DraftKings' lobby lists a man once per DRAFTABLE entry, so a player eligible
at two positions arrives twice. Twice on the board means his ownership is
counted twice and a lineup can roster him twice, which is not a legal entry.

The obvious fix - collapse rows with the same name - is wrong, and wrong in
the more expensive direction. There are two Max Muncys in this league, on
different teams. Deduping by name deletes one of them, and a missing player
leaves no row to be wrong about: nothing downstream can detect him, the same
way three players missing from the NFL showdown board were invisible to every
measurement until a finished contest was joined against it.

So the rule is: same DraftKings id means same man; same name means nothing.
"""
from __future__ import annotations

import sys

import pandas as pd


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """The dedup exactly as `mlb_data.board` performs it."""
    ids = pd.to_numeric(df["dk_player_id"], errors="coerce")
    if ids.notna().any():
        return (df[ids.notna()]
                .drop_duplicates("dk_player_id", keep="first")
                .reset_index(drop=True))
    return df.reset_index(drop=True)


def row(pid, name, team, pos, salary=4000):
    return {"dk_player_id": pid, "name": name, "team": team,
            "position": pos, "salary": salary}


def a_multi_position_player_collapses_to_one_row():
    """The actual bug: one man, two eligibilities, two rows."""
    df = pd.DataFrame([
        row(1, "Max Muncy", "LAD", "3B"),
        row(1, "Max Muncy", "LAD", "1B"),
        row(2, "Mookie Betts", "LAD", "OF"),
    ])
    out = collapse(df)
    assert len(out) == 2, f"expected 2 rows, got {len(out)}"
    assert (out["name"] == "Max Muncy").sum() == 1


def two_different_players_with_one_name_both_survive():
    """The Max Muncy problem. Same name, different men, different teams.

    A name-based dedup passes the test above and fails this one, which is
    exactly why this test exists.
    """
    df = pd.DataFrame([
        row(1, "Max Muncy", "LAD", "3B"),
        row(2, "Max Muncy", "ATH", "2B"),
    ])
    out = collapse(df)
    assert len(out) == 2, (
        "a real player was deleted - these are two different men who share a "
        "name, and only their ids say so")
    assert set(out["team"]) == {"LAD", "ATH"}


def the_first_listing_is_the_one_kept():
    """DraftKings lists the primary eligibility first, and the optimiser
    pins a player to his first position, so keeping the first row keeps the
    two consistent."""
    df = pd.DataFrame([
        row(7, "Somebody", "NYM", "SS"),
        row(7, "Somebody", "NYM", "2B"),
    ])
    out = collapse(df)
    assert out["position"].iloc[0] == "SS"


def a_row_without_an_id_is_dropped():
    """It cannot be deduped and it cannot be joined to a projection either.

    Keeping it would put a player on the board who can never carry a number,
    which is a silent hole rather than a visible gap.
    """
    df = pd.DataFrame([
        row(1, "Has An Id", "LAD", "1B"),
        row(None, "No Id At All", "LAD", "OF"),
    ])
    out = collapse(df)
    assert len(out) == 1
    assert out["name"].iloc[0] == "Has An Id"


def a_board_with_no_ids_at_all_is_left_alone():
    """Better a board with duplicates than an empty one.

    If DraftKings renames the id field, dropping every row without an id
    would delete the entire slate. The publisher logs an error instead.
    """
    df = pd.DataFrame([
        row(None, "A", "LAD", "1B"),
        row(None, "B", "LAD", "OF"),
    ])
    assert len(collapse(df)) == 2


def nothing_changes_on_a_clean_board():
    df = pd.DataFrame([row(i, f"P{i}", "LAD", "OF") for i in range(1, 11)])
    assert len(collapse(df)) == 10


SUITES = [
    ("ONE ROW PER PLAYER", [
        ("a multi-position player collapses to one row",
         a_multi_position_player_collapses_to_one_row),
        ("the first listing is the one kept",
         the_first_listing_is_the_one_kept),
        ("a clean board is untouched", nothing_changes_on_a_clean_board),
    ]),
    ("SAME NAME IS NOT SAME PLAYER", [
        ("two Max Muncys both survive",
         two_different_players_with_one_name_both_survive),
    ]),
    ("A PLAYER WITH NO ID", [
        ("is dropped, because he can never be joined",
         a_row_without_an_id_is_dropped),
        ("unless NOBODY has an id, when the board is left alone",
         a_board_with_no_ids_at_all_is_left_alone),
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
    sys.exit(main())
