"""End to end, against live data, before any modelling is built on top.

The unit tests prove the logic is right about cases I thought of. This proves
the data layer is right about this Saturday, which is a different claim and
the one that matters. It runs the same path the real build will run and
reports what it found rather than whether it crashed - a pipeline that
completes while silently matching nobody looks identical to a working one from
the outside.

The load-bearing check is the scoring validation. DraftKings publishes each
player's own points per game on the board. Recomputing that number from CFBD
stats and comparing is the only end-to-end test available: it exercises the
stat map, the scoring rules, the bonuses, the name join and the season filter
at once, against a number this project did not produce. If it agrees to within
a tenth of a point, everything upstream of it is right. If it does not, one of
those five things is wrong and the difference says which.

The comparison is restricted to players whose game count matches their team's.
Neither `mean` nor `points / team games` matches DraftKings universally -
players who missed games are divided by something else - and grading on the
whole board produced a confident 2.00x and then a confident 0.50x before that
was understood.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import cfb_data as D

VERSION = "v1"
log = logging.getLogger("cfb_verify")


def _key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set. Add it as a repository secret.")
    return key


def head(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def pick_slate(sl: pd.DataFrame, want: str) -> pd.Series:
    """The busiest slate of the requested type, by contest count."""
    pool = sl[sl["game_type"].astype(str).str.contains(want, case=False,
                                                       na=False)]
    if pool.empty:
        log.warning("no %r slate; falling back to the busiest of any type",
                    want)
        pool = sl
    return pool.iloc[0]


def scoring_check(board: pd.DataFrame, hist: pd.DataFrame) -> int:
    """Recompute DraftKings' own points per game and compare. 0 if sane."""
    head("SCORING VALIDATION  (ours vs DraftKings' published ppg)")

    matched = board[board["athlete_id"].notna()].copy()
    if matched.empty or "dk_points_per_game" not in matched:
        print("no matched players with a published ppg - cannot validate")
        return 1

    per = (hist.groupby("athlete_id")
           .agg(total=("points", "sum"), games=("points", "size"))
           .reset_index())
    team_games = (hist.drop_duplicates(["school", "week"])
                  .groupby("school").size().rename("team_games"))
    school = hist.drop_duplicates("athlete_id").set_index("athlete_id")["school"]
    per["school"] = per["athlete_id"].map(school)
    per = per.merge(team_games, left_on="school", right_index=True, how="left")

    m = matched.merge(per, on="athlete_id", how="inner")
    m = m[m["dk_points_per_game"].notna() & (m["dk_points_per_game"] > 0)]
    if m.empty:
        print("nothing to compare")
        return 1

    full = m[m["games"] == m["team_games"]].copy()
    if len(full) < 20:
        print(f"only {len(full)} players played every team game - too few to "
              f"validate cleanly; comparing all {len(m)} instead")
        full = m.copy()

    full["ours"] = full["total"] / full["games"]
    full["diff"] = full["ours"] - full["dk_points_per_game"]
    mae = full["diff"].abs().mean()
    bias = full["diff"].mean()
    ratio = (full["ours"] / full["dk_points_per_game"]).median()

    print(f"players compared : {len(full)} (played all {int(full['team_games'].median())} team games)")
    print(f"MAE              : {mae:.3f} points per game")
    print(f"bias             : {bias:+.3f}")
    print(f"median ratio     : {ratio:.4f}   (1.0000 is exact agreement)")
    print()
    worst = full.reindex(full["diff"].abs().sort_values(ascending=False).index)
    print("largest disagreements:")
    print(f"  {'player':<26}{'pos':<5}{'ours':>8}{'DK':>8}{'diff':>8}")
    for r in worst.head(8).itertuples(index=False):
        print(f"  {str(r.name)[:25]:<26}{str(r.position):<5}"
              f"{r.ours:>8.2f}{r.dk_points_per_game:>8.2f}{r.diff:>+8.2f}")

    print()
    if mae < 0.15:
        print("PASS - the stat map, scoring rules, bonuses and join all agree")
        print("with a number this project did not produce.")
        return 0
    if 1.4 < ratio < 1.6 or 0.6 < ratio < 0.72:
        print("FAIL - a ratio near 1.5 or 0.67 is a captain multiplier applied")
        print("to the wrong side, not a scoring error.")
        return 1
    if 1.9 < ratio < 2.1 or 0.45 < ratio < 0.55:
        print("FAIL - a ratio near 2 or 0.5 is a divisor problem: the wrong")
        print("game count, not the wrong scoring.")
        return 1
    print("FAIL - scoring disagrees with DraftKings by more than a tenth of a")
    print("point per game. Do not build projections on this.")
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--first-season", type=int, default=2023,
                   help="earliest season of training history to join against")
    p.add_argument("--draft-group", type=int, default=0)
    p.add_argument("--game-type", default="Classic")
    args = p.parse_args()

    print(f"cfb_verify {VERSION}")
    key = _key()
    problems = 0

    head("UNIT TESTS")
    import test_cfb_data
    if test_cfb_data.main() != 0:
        print("\nunit tests failed - stopping before touching live data")
        return 1

    head("SLATES ON SALE")
    sl = D.slates()
    print(sl.head(15).to_string(index=False))

    row = (sl[sl["draft_group"] == args.draft_group].iloc[0]
           if args.draft_group else pick_slate(sl, args.game_type))
    dg = int(row["draft_group"])
    is_showdown = "showdown" in str(row["game_type"]).lower()
    print(f"\nusing draft group {dg} - {row['game_type']} - "
          f"{row['contests']} contests - starts {row['starts_text']}")

    head("THE BOARD")
    board = D.board(dg, captain_multiplier=1.5 if is_showdown else None)
    print(board[["name", "position", "team", "opponent", "salary",
                 "charged_salary", "is_captain",
                 "dk_points_per_game"]].head(15).to_string(index=False))
    print(f"\npositions: {board['position'].value_counts().to_dict()}")
    print(f"salary   : ${board['salary'].min():,.0f} - "
          f"${board['salary'].max():,.0f}")
    if is_showdown:
        caps = int(board["is_captain"].sum())
        print(f"captains : {caps} rows charged 1.5x")
        if caps == 0:
            print("  WARNING: a showdown slate with no captain rows detected")
            problems += 1

    head("LIVE WEEK")
    week = D.live_week(key, args.season)
    print(f"newest completed week of {args.season}: {week}")

    head("TEAM MAP")
    games = D.upcoming_games(key, args.season, week, span=2)
    # The teams endpoint carries abbreviations and alternate names. Without it
    # this mapped 6 of 24: Georgia's abbreviation IS "UGA", and no rule
    # derives that from the string "Georgia".
    teams = D.cfbd("teams", key, year=args.season)
    mapping = D.fixture_team_map(board, games, teams)
    codes = sorted(set(board["team"].dropna()))
    for code in codes:
        print(f"  {code:<8} -> {mapping.get(code, '*** UNSOLVED ***')}")
    missing = [c for c in codes if c not in mapping]
    print(f"\n{len(codes) - len(missing)} of {len(codes)} codes solved")
    if missing:
        print(f"UNSOLVED: {missing}")
        problems += 1

    head("HISTORY")
    # The first run joined against two weeks of one season and matched 44% of
    # the board, which looked like a broken join and was not: a backup
    # quarterback priced at $8,500 with 0.0 published points per game has no
    # 2026 stats because he has not played. The model trains on many seasons,
    # so the join has to be measured against many seasons.
    hist = D.season_history(key, args.season, week)
    print(f"{len(hist):,} player-games, {hist['athlete_id'].nunique():,} "
          f"athletes, weeks 1-{week}")
    print(f"points: {hist['points'].min():.1f} to {hist['points'].max():.1f}, "
          f"mean {hist['points'].mean():.2f}")
    print("\nby position:")
    print(hist[hist["position"].isin(D.SKILL_POSITIONS)]
          .groupby("position")["points"]
          .agg(["size", "mean", "max"]).round(2).to_string())

    head("THE JOIN  (current season only)")
    joined = D.attach_history(board, hist)
    print(D.join_quality(joined))
    print("\nThis number is expected to be poor early in a season and is NOT")
    print("the one to judge. Backups with no snaps have no stats. The")
    print("multi-season join below is what the model actually sees.")

    # Scoring is validated against the CURRENT season only, because
    # DraftKings' published points per game is a this-season number.
    problems += scoring_check(joined, hist)

    head(f"THE JOIN  (training history, {args.first_season}-{args.season})")
    deep = D.training_history(key, args.first_season, args.season,
                              through_week={args.season: week})
    print(f"{len(deep):,} player-games across "
          f"{deep['season'].nunique()} seasons, "
          f"{deep['athlete_id'].nunique():,} athletes")
    joined_deep = D.attach_history(board, deep)
    print()
    print(D.join_quality(joined_deep))
    rate = joined_deep["athlete_id"].notna().mean()
    print()
    if rate >= 0.90:
        print(f"PASS - {rate:.1%} of the priced board has history to project "
              f"from.")
    else:
        print(f"FAIL - only {rate:.1%} of the board has any history. "
              f"Projections would be guesses for the rest.")
        problems += 1

    head("VERDICT")
    if problems == 0:
        print("Data layer verified against live data. Safe to build the")
        print("model on top of it.")
    else:
        print(f"{problems} problem(s) above. Fix before modelling - every")
        print("one of them produces confident numbers rather than an error.")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
