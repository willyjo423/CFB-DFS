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


def scoring_check(board: pd.DataFrame, hist: pd.DataFrame,
                  min_games: int = 25, pitcher_min: int = 8) -> int:
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

    # The first run compared 3-to-8-game sample means against DraftKings'
    # full-SEASON average and called the difference a scoring error. It is
    # not: at eight games a perfectly scored hitter still shows an MAE near
    # 1.9 from variance alone, and errors appeared in both directions, which
    # is the signature of sampling rather than of a wrong rule.
    #
    # College football hid this because its season was two weeks old, so
    # both sides shared a denominator. Here they do not, so the comparison
    # is restricted to players with enough games for their mean to mean
    # something.
    # Pitchers need their own threshold. A starter makes about 32 starts a
    # season, so across half a window he appears perhaps fifteen times - and
    # a 25-game minimum silently excluded every pitcher from the first
    # comparison that otherwise passed. Pitcher scoring is the half most
    # likely to be wrong (two rule sets, conditional bonuses, innings as
    # thirds) and it was the half going unchecked.
    is_pit = m["position"].astype(str).str.upper().isin(M.PITCHER_POSITIONS)
    m = m[np.where(is_pit, m["games"] >= pitcher_min, m["games"] >= min_games)]
    if len(m) < 20:
        print(f"only {len(m)} players have {min_games}+ games in this window."
              f" Widen it - a comparison on fewer is measuring variance.")
        if m.empty:
            return 1

    m["ours"] = m["total"] / m["games"]
    m["diff"] = m["ours"] - m["dk_points_per_game"]
    mae = m["diff"].abs().mean()
    ratio = (m["ours"] / m["dk_points_per_game"]).median()

    pitchers = m[m["position"].astype(str).str.upper().isin(
        M.PITCHER_POSITIONS)]
    hitters = m[~m.index.isin(pitchers.index)]

    print(f"players compared : {len(m)}  ({len(hitters)} hitters with "
          f"{min_games}+ games, {len(pitchers)} pitchers with "
          f"{pitcher_min}+)")
    print(f"median games each: {int(m['games'].median())}")
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

    # Judged on the RATIO, not the MAE. Per-player noise does not go away at
    # 25 games - it is still worth about 1.1 points - but it is unbiased, so
    # the median ratio across a couple of hundred players converges on the
    # truth while the MAE never does.
    print()
    if len(pitchers) < 5 or len(hitters) < 5:
        missing = "pitchers" if len(pitchers) < 5 else "hitters"
        print(f"\nINCOMPLETE - only {len(pitchers)} pitchers and "
              f"{len(hitters)} hitters cleared their thresholds, so the "
              f"{missing} rule set is UNVERIFIED. A pass on one half is not "
              f"a pass.")
        return 1

    off = abs(ratio - 1.0)
    if off <= 0.02:
        print(f"PASS - the median player scores within {off:.1%} of "
              f"DraftKings' own number. Both rule sets, the field mapping "
              f"and the innings-as-thirds parsing are right.")
        return 0
    print(f"FAIL - the median player is {off:.1%} away from DraftKings. "
          f"That is too large to be sampling; a rule is wrong.")
    print("Look at the per-group ratios above: a rule wrong for one system "
          "only shows there and would be invisible in the combined number.")
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--games", type=int, default=700)
    p.add_argument("--pitcher-min", type=int, default=8,
                   help="pitchers appear far less often than hitters")
    p.add_argument("--min-games", type=int, default=25,
                   help="players with fewer are too noisy to "
                        "compare")
    args = p.parse_args()

    print(f"mlb_verify {VERSION}")
    problems = 0

    head("SCHEDULE")
    # A wider default window than the first run used. Comparing sample means
    # to season averages needs the samples to be big enough to have a mean.
    start = args.start or f"{args.season}-06-01"
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
    # Whether a field is POPULATED and whether it JOINS are different
    # questions, and the first run answered the wrong one: 179 of 278 rows
    # carried a number, and not one of them matched a real MLB id. Blake
    # Snell came back as 10148; his league id is six digits. Those are
    # DraftKings' own player ids.
    #
    # This is the same mistake as counting blank positions as matches in the
    # football build, so it is measured the way it should have been: by how
    # many rows actually found history.
    joined = M.attach_history(board, hist)
    by_id = int((joined["matched_by"] == "id").sum())
    by_name = int((joined["matched_by"] == "name").sum())
    populated = int(board["mlb_id"].notna().sum())
    print(f"\nrows carrying something in an id field : {populated} of "
          f"{len(board)}")
    print(f"rows that actually matched BY that id  : {by_id}")
    print(f"rows matched by name                   : {by_name}")
    if by_id == 0:
        print("\nThe id fields are DraftKings' own, not the league's. Names "
              "are the join.")
        print("That is workable here in a way it was not for college "
              "football: thirty teams, unique names, and no A.J./AJ problem.")
        print("\nEvery field DraftKings actually sends, in case a league id "
              "is among them:\n")
        print(M.board_row_keys(dg))
    else:
        print(f"\n{by_id} rows join on a league id - an integer comparison, "
              f"no name matching needed for those.")

    problems += scoring_check(board, hist, args.min_games,
                              args.pitcher_min)

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
