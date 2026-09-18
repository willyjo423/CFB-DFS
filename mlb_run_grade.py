"""Grade the baseball model, walk-forward, one half of the sport at a time.

This is the file that decides whether any of the rest is worth shipping. For
each graded day it fits on everything strictly before that day and predicts
it - never the other way round - and reports against three baselines rather
than in isolation, because an MAE means nothing on its own.

Hitters and pitchers are graded SEPARATELY and neither result is allowed to
hide behind the other. They are different sports with different variance:
pitchers swing from -20 to +58 in a single outing, hitters from 0 to 59, and
a combined number would let a good half carry a bad one. The MLB scoring
check already made that mistake once - it printed PASS while zero pitchers
had been compared.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import mlb_sport as MS
import mlb_statcast as MSC
from engine import cache as C
from engine import features as EF
from engine import grade as G

VERSION = "v1"
SPORT = "mlb"
log = logging.getLogger("mlb_run_grade")


def head(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def grade_half(built: pd.DataFrame, which: str, test_season: int,
               day: int) -> tuple[pd.DataFrame, bool]:
    """Grade one calendar day.

    A day is `MS.SLOTS` periods rather than one, because a doubleheader is
    two games, two lineups and two roster decisions - and grading only the
    first one would quietly drop the days with the most opportunity on them.
    """
    spec = MS.SPECS[which]
    first, last = MS.periods_of(day)
    head(f"{which.upper()}  ({spec.name})  day {day}")

    tr = EF.trainable(built, spec)
    print(f"{len(built):,} rows, {len(tr):,} trainable "
          f"(>= {spec.min_prior_games} prior appearances)")
    print(f"{len(spec.features)} features, half-life {spec.halflife} periods")
    if len(tr) < spec.min_train_rows:
        print(f"NOT GRADEABLE: {len(tr)} trainable rows, need "
              f"{spec.min_train_rows}")
        return pd.DataFrame(), False

    print()
    print(G.prove_no_leak(built, spec, test_season, first))

    try:
        graded = G.walk(built, spec, test_season, first, last)
    except ValueError as exc:
        print(f"\nNOT GRADEABLE: {exc}")
        return pd.DataFrame(), False

    print(f"\n{len(graded):,} player-days graded")
    print(G.report(graded, spec))

    acc = G.accuracy(graded).set_index("key")
    ok = True
    print()
    if "mean" in acc.index:
        mine = acc.loc["mean", "MAE"]
        for base in ("prior_mean", "prior_last", "prior_position"):
            if base in acc.index:
                theirs = acc.loc[base, "MAE"]
                better = bool(mine < theirs)
                ok = ok and better
                print(f"  beats {base:<16}: {better}   "
                      f"({mine:.3f} vs {theirs:.3f})")
    return graded, ok


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--first-season", type=int, default=2024)
    p.add_argument("--test-season", type=int, default=2026)
    p.add_argument("--first-day", type=int, default=182,
                   help="day of year to start grading (182 = 1 July)")
    p.add_argument("--last-day", type=int, default=258,
                   help="day of year to stop (258 = 15 September)")
    p.add_argument("--every", type=int, default=7,
                   help="grade every Nth day; every day is 70+ refits")
    p.add_argument("--quantiles", default="",
                   help="comma separated, e.g. 0.1,0.5,0.9 to run faster")
    args = p.parse_args()

    print(f"mlb_run_grade {VERSION}")

    if args.quantiles:
        import dataclasses
        qs = [float(x) for x in args.quantiles.split(",")]
        for k in list(MS.SPECS):
            MS.SPECS[k] = dataclasses.replace(MS.SPECS[k], quantiles=qs)
        MS.HITTERS, MS.PITCHERS = MS.SPECS["hitters"], MS.SPECS["pitchers"]
        print(f"quantiles overridden: {qs}")

    head("HISTORY")
    seasons = list(range(args.first_season, args.test_season + 1))
    hist = C.load(SPORT, seasons)
    # Optional. An absent Statcast cache leaves those columns missing and
    # grades the model as it was, which is exactly what a comparison run
    # needs to be able to do.
    statcast = MSC.load(seasons)
    have = sorted(int(s) for s in
                  pd.to_datetime(hist["date"], errors="coerce").dt.year
                  .dropna().unique())
    print(f"{len(hist):,} player-games, seasons {have}")
    if args.test_season not in have:
        sys.exit(f"the season being graded ({args.test_season}) is not "
                 f"cached; run the MLB fetch workflow first")

    # Grading every day is 77 refits per half. Weekly is 11, which is plenty
    # to measure calibration and ranking and finishes in a sitting.
    # Say out loud what the doubleheader packing did, rather than trusting it.
    canon = MS.to_canonical(hist)
    second = canon[canon["period"] % MS.SLOTS > 0]
    dupes = int(canon.duplicated(["player_id", "season", "period"]).sum())
    print(f"doubleheaders: {second['game_pk'].nunique():,} second games, "
          f"{len(second):,} player-rows")
    print(f"duplicate player-period keys after packing: {dupes}")
    if dupes:
        bad = canon[canon.duplicated(["player_id", "season", "period"],
                                     keep=False)]
        print(bad[["name", "date", "team", "opponent", "game_pk",
                   "period", "points"]].head(12).to_string(index=False))
        sys.exit("period still does not identify a game - see the rows above")

    days = list(range(args.first_day, args.last_day + 1, args.every))
    print(f"grading {len(days)} days: {days[0]} to {days[-1]} "
          f"every {args.every}")

    results = {}
    for which in ("hitters", "pitchers"):
        # Features are built ONCE per half, not once per graded day. They do
        # not depend on which day is being graded - only the fit does, and
        # walk() re-fits per day on its own. Rebuilding here was 22 identical
        # passes over 140,000 rows.
        built = MS.build(hist, which, statcast)
        frames = []
        allok = True
        for d in days:
            g, ok = grade_half(built, which, args.test_season, d)
            if len(g):
                frames.append(g)
            allok = allok and ok
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(f"mlb_graded_{which}.csv", index=False)
            results[which] = (combined, allok, len(frames))

    head("VERDICT")
    if not results:
        print("Nothing was gradeable. Check the cache and the day range.")
        return 1

    # Judged on the POOLED rows, not on every single day.
    #
    # The per-day flag ANDs together ten independent comparisons of about 300
    # rows each, so one day losing by 0.001 of an MAE point condemns the whole
    # half. That is a coin-flip masquerading as a verdict. The pooled number
    # is the one with the sample size behind it; the day count is reported
    # beside it because a model that wins narrowly nine times out of ten is a
    # different animal from one that wins hugely twice and loses eight times.
    overall = True
    for which, (g, allok, n_days) in results.items():
        spec = MS.SPECS[which]
        acc = G.accuracy(g).set_index("key")
        mine = acc.loc["mean", "MAE"]
        cal = G.calibration(g, spec)["error"].abs().sum()
        av = G.availability(g)

        print(f"\n  {which.upper()}  -  {len(g):,} rows over {n_days} days")
        beats_all = True
        for base, label in (("prior_mean", "his own average so far"),
                            ("prior_last", "what he did last time"),
                            ("prior_position", "the position average")):
            if base not in acc.index:
                continue
            theirs = acc.loc[base, "MAE"]
            better = bool(mine < theirs)
            beats_all = beats_all and better
            edge = (theirs - mine) / theirs * 100.0
            print(f"    vs {label:<24} {mine:6.3f} vs {theirs:6.3f}"
                  f"   {edge:+5.1f}%   {'beats' if better else 'LOSES'}")
        print(f"    every day individually : {allok}")
        print(f"    calibration error      : {cal:.3f}")
        if av:
            print(f"    availability AUC       : {av['AUC']:.3f}")
        overall = overall and beats_all

    print()
    if overall:
        print("Both halves beat every baseline on the pooled rows.")
        print("Read the per-baseline percentages above before celebrating: an")
        print("edge under about 1% over a player's own running average is not")
        print("a projection, it is that average with extra steps.")
    else:
        print("At least one half does NOT beat every baseline. A projection")
        print("that loses to 'what he did last time' is not a projection -")
        print("read the tables above before building anything on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
