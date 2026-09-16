"""Grade the CFB model against real college football, walk-forward.

This is the file that decides whether any of the rest is worth shipping. It
pulls real history, builds features, and for each graded week fits on
everything strictly before it and predicts it - never the other way round.

It reports against three baselines rather than in isolation, because an MAE
of 4.5 means nothing on its own, and it runs a leak proof that is capable of
failing. If the model cannot beat "what he did last week", nothing downstream
of it deserves a page.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

import cfb_data as D
import cfb_features as F
import cfb_grade as G
import cfb_model as M

VERSION = "v1"
log = logging.getLogger("cfb_run_grade")


def _key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set. Add it as a repository secret.")
    return key


def head(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--first-season", type=int, default=2021)
    p.add_argument("--test-season", type=int, default=2025)
    p.add_argument("--first-week", type=int, default=6)
    p.add_argument("--last-week", type=int, default=13)
    p.add_argument("--quantiles", default="",
                   help="comma separated, e.g. 0.1,0.5,0.9 to run faster")
    args = p.parse_args()

    print(f"cfb_run_grade {VERSION}")
    key = _key()

    if args.quantiles:
        M.QUANTILES = [float(x) for x in args.quantiles.split(",")]
        print(f"quantiles overridden: {M.QUANTILES}")

    head("HISTORY")
    # Positions are REQUIRED here, unlike for the join: the model uses
    # position as a feature, grading is restricted by position, and the
    # position baseline is computed per position. A row without one cannot
    # take part in any of that.
    hist = D.training_history(key, args.first_season, args.test_season,
                              require_position=True)
    print(f"{len(hist):,} player-games, {hist['athlete_id'].nunique():,} "
          f"athletes, seasons {sorted(hist['season'].unique())}")

    head("FEATURES")
    built = F.build(hist)
    tr = F.trainable(built)
    print(f"{len(built):,} rows built, {len(tr):,} trainable "
          f"(>= {F.MIN_PRIOR_GAMES} prior games, skill positions only)")
    print(f"{len(F.FEATURES)} features")
    played = (tr["points"] > 0).mean()
    print(f"{played:.1%} of trainable rows involve a player who actually "
          f"recorded something")

    head("LEAK PROOF")
    print(G.prove_no_leak(built, args.test_season, args.first_week))

    head(f"WALK-FORWARD  {args.test_season} weeks "
         f"{args.first_week}-{args.last_week}")
    graded = G.walk(built, args.test_season, args.first_week, args.last_week)
    print(f"{len(graded):,} player-weeks graded")

    print(G.report(graded))

    head("VERDICT")
    acc = G.accuracy(graded).set_index("key")
    ok = True
    if "mean" in acc.index:
        mine = acc.loc["mean", "MAE"]
        for base in ("prior_mean", "prior_last", "prior_position"):
            if base in acc.index:
                theirs = acc.loc[base, "MAE"]
                better = mine < theirs
                ok = ok and better
                print(f"  beats {base:<16}: {better}   "
                      f"({mine:.3f} vs {theirs:.3f})")
    cal = G.calibration(graded)["error"].abs().sum()
    print(f"  calibration error   : {cal:.3f}  (0.05 or less is good)")
    av = G.availability(graded)
    if av:
        print(f"  availability AUC    : {av['AUC']:.3f}")
        print(f"  sharpness           : {av['sharpness']:.3f}")

    graded.to_csv("cfb_graded.csv", index=False)
    print("\nwrote cfb_graded.csv")

    print()
    if ok:
        print("The model beats every baseline. Worth building the simulator,")
        print("optimiser and page on top of.")
    else:
        print("The model does NOT beat every baseline. Read the table above")
        print("before building anything on it - a projection that loses to")
        print("'what he did last week' is not a projection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
