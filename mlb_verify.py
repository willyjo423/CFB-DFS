"""Everything unknown about baseball, answered in one run.

The college football build spent five separate runs establishing facts one at
a time - how deep the history goes, whether the current season is available,
whether ids join, whether the scoring is right. That was too slow and the
questions were never independent.

So this asks all of them at once and reports what it found rather than
whether it crashed:

  1. Does the schedule endpoint answer, and how many games are final?
  2. What does a box score ACTUALLY contain? Printed verbatim, because this
     project's scoring rules are asserted from memory and the field names
     are not. A missing `hitByPitch` is worth knowing before it becomes "the
     numbers look slightly off".
  3. Do the two scoring systems reproduce DraftKings' own published points
     per game? This is the decisive check - it exercises the field mapping,
     both rule sets, the innings-as-thirds parsing and the join at once,
     against a number this project did not produce.
  4. Does DraftKings carry the MLB player id? If it does the join is an
     integer comparison and none of college football's name misery repeats.
     If it does not, the fallback is names and this says so loudly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

import mlb_data as M

VERSION = "v1"
log = logging.getLogger("mlb_verify")


def head(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def gather(season: int, start: str, end: str, limit: int) -> pd.DataFrame:
    """Box scores for a window of finished games."""
    sched = M.schedule(season, start, end)
    final = sched[sched["final"]].head(limit)
    print(f"{len(sched)} games in window, {int(sched['final'].sum())} final, "
          f"pulling {len(final)}")
    rows, shown = [], False
    for i, g in enumerate(final.to_dict("records"), 1):
        try:
            box = M.boxscore(g["game_pk"])
        except M.Unavailable as exc:
            log.warning("game %s unavailable: %s", g["game_pk"], exc)
            continue
        if not shown:
            head("WHAT A BOX SCORE ACTUALLY CONTAINS")
            print(M.describe_payload(box))
            shown = True
        rows += M.player_games(box, g)
        if i % 25 == 0:
            print(f"  ... {i} games, {len(rows)} player-games")
        time.sleep(0.12)
    if not rows:
        sys.exit("no player rows were built; nothing further can be checked")
    return M.score(pd.DataFrame(rows))


def scoring_check(board: pd.DataFrame, hist: pd.DataFrame) -> int:
    """Recompute DraftKings' published ppg from our rules. 0 if sane."""
    head("SCORING VALIDATION  (ours vs DraftKings' published ppg)")
    joined = M.attach_history(board, hist)
    m = joined[joined["player_id"].notna()].copy()
    m = m[pd.to_numeric(m["dk_points_per_game"],
                        errors="coerce").fillna(0) > 0]
    if m.empty:
        print("nothing matched with a published ppg - cannot validate")
        return 1

    per = (hist.groupby("player_id")
           .agg(total=("points", "sum"), games=("points", "size"))
           .reset_index())
    m = m.merge(per, on="player_id", how="inner")
    m = m[m["games"] >= 3]
    if len(m) < 20:
        print(f"only {len(m)} players have three or more games in this "
              f"window - widen it before trusting this")
        if m.empty:
            return 1

    m["ours"] = m["total"] / m["games"]
    m["diff"] = m["ours"] - m["dk_points_per_game"]
    mae = m["diff"].abs().mean()
    ratio = (m["ours"] / m["dk_points_per_game"]).median()

    pitchers = m[m["position"].astype(str).str.upper().isin(
        M.PITCHER_POSITIONS)]
    hitters = m[~m.index.isin(pitchers.index)]

    print(f"players compared : {len(m)}  "
          f"({len(hitters)} hitters, {len(pitchers)} pitchers)")
    print(f"MAE              : {mae:.3f} points per game")
    print(f"median ratio     : {ratio:.4f}   (1.0000 is exact agreement)")
    for label, sub in (("hitters", hitters), ("pitchers", pitchers)):
        if len(sub) >= 5:
            print(f"  {label:<9}: MAE {sub['diff'].abs().mean():.3f}, "
                  f"median ratio "
                  f"{(sub['ours'] / sub['dk_points_per_game']).median():.4f}")

    print()
    worst = m.reindex(m["diff"].abs().sort_values(ascending=False).index)
    print(f"  {'player':<24}{'pos':<5}{'g':>4}{'ours':>8}{'DK':>8}{'diff':>8}")
    for r in worst.head(10).itertuples(index=False):
        print(f"  {str(r.name)[:23]:<24}{str(r.position):<5}{int(r.games):>4}"
              f"{r.ours:>8.2f}{r.dk_points_per_game:>8.2f}{r.diff:>+8.2f}")

    print()
    if mae < 0.75:
        print("PASS - both scoring systems, the field mapping and the "
              "innings-as-thirds parsing agree with a number this project "
              "did not produce.")
        return 0
    print("FAIL - scoring disagrees with DraftKings. The box-score keys "
          "printed above are the place to look; a rule that is wrong for "
          "one system only will show in the per-group MAEs.")
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--games", type=int, default=120)
    args = p.parse_args()

    print(f"mlb_verify {VERSION}")
    problems = 0

    head("SCHEDULE")
    start = args.start or f"{args.season}-08-15"
    end = args.end or f"{args.season}-09-15"
    print(f"window: {start} to {end}")
    hist = gather(args.season, start, end, args.games)

    head("HISTORY")
    print(f"{len(hist):,} player-games, "
          f"{hist['player_id'].nunique():,} players")
    pit = hist[hist["is_pitcher"] == 1]
    hit = hist[hist["is_pitcher"] == 0]
    print(f"  hitters : {len(hit):,} rows, points "
          f"{hit['points'].min():.1f} to {hit['points'].max():.1f}, "
          f"mean {hit['points'].mean():.2f}")
    print(f"  pitchers: {len(pit):,} rows, points "
          f"{pit['points'].min():.1f} to {pit['points'].max():.1f}, "
          f"mean {pit['points'].mean():.2f}")
    print(f"  wins credited: {int(hist['win'].sum())}  "
          f"(should be about one per game)")
    if hist["win"].sum() == 0:
        print("  WARNING: no wins found. The decisions block is elsewhere in "
              "this payload shape and pitchers are losing 4 points each.")
        problems += 1
    print(f"\npositions seen: "
          f"{sorted(hist['position'].dropna().astype(str).unique())}")

    head("SLATES ON SALE")
    try:
        sl = M.slates()
        print(sl.head(10).to_string(index=False))
        dg = int(sl.iloc[0]["draft_group"])
    except M.Unavailable as exc:
        print(f"no slates: {exc}")
        print("\nOut of season, the lobby is empty and the scoring check "
              "below cannot run. Everything above still stands.")
        return problems

    head("THE BOARD")
    board = M.board(dg)
    print(board[["name", "position", "team", "opponent", "salary",
                 "mlb_id", "dk_points_per_game"]].head(12).to_string(
                     index=False))
    with_id = int(board["mlb_id"].notna().sum())
    print(f"\nDraftKings rows carrying an MLB id: {with_id} of {len(board)} "
          f"({100 * with_id / max(1, len(board)):.0f}%)")
    if with_id == 0:
        print("NONE. The join falls back to names - workable, but this is "
              "the thing that made college football slow, so it is worth "
              "finding the right field before building on it.")
        problems += 1
    else:
        print("The join is an integer comparison. None of college "
              "football's name misery repeats here.")

    problems += scoring_check(board, hist)

    head("VERDICT")
    if problems == 0:
        print("Baseball's data layer is verified. Safe to build the spec and "
              "grade the model on it.")
    else:
        print(f"{problems} problem(s) above - each one produces confident "
              f"numbers rather than an error, so fix before modelling.")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
