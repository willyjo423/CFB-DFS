"""Download seasons of box scores into the cache, once, and stop.

A season is about 2,430 games and each needs its own box score, so this is
roughly twelve minutes of HTTP per season. That is exactly why it is cached
and committed rather than paid on every run.

Deliberately conservative about failure, because the college football build
lost five minutes of downloading to a quota error that discarded everything
fetched before it:

  * a season already on disk is skipped unless explicitly refreshed;
  * each season is written as soon as it completes, not at the end, so a
    failure on season three keeps seasons one and two;
  * a run that dies partway leaves the cache strictly better than it found
    it, and re-running picks up only what is missing.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

import mlb_data as M
from engine import cache as C

SPORT = "mlb"


def fetch_season(season: int, limit: int = 0, pause: float = 0.10
                 ) -> pd.DataFrame:
    """Every finished game of a season, as scored player-rows."""
    sched = M.schedule(season)
    done = sched[sched["final"]]
    if limit:
        done = done.head(limit)
    print(f"{season}: {len(sched)} games, {len(done)} final, pulling them")

    rows = []
    for i, g in enumerate(done.to_dict("records"), 1):
        try:
            rows += M.player_games(M.boxscore(g["game_pk"]), g)
        except M.Unavailable as exc:
            logging.warning("game %s unavailable: %s", g["game_pk"], exc)
            continue
        if i % 200 == 0:
            print(f"  ... {i}/{len(done)} games, {len(rows):,} player-games")
        time.sleep(pause)
    if not rows:
        raise RuntimeError(f"{season}: no player rows were built")
    df = M.score(pd.DataFrame(rows))
    print(f"{season}: {len(df):,} player-games, "
          f"{df['player_id'].nunique():,} players")
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2024 2025 2026")
    p.add_argument("--refresh", default="",
                   help="seasons to re-fetch even if cached (the live one)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap games per season, for a quick trial")
    args = p.parse_args()

    seasons = [int(x) for x in args.seasons.split()]
    refresh = {int(x) for x in args.refresh.split()} if args.refresh else set()

    print("cached before:", C.cached_seasons(SPORT) or "nothing")
    print("requested    :", seasons)
    print("refreshing   :", sorted(refresh) or "nothing")
    print()

    done, skipped, failed = [], [], []
    for season in sorted(seasons):
        if C.have(SPORT, season) and season not in refresh:
            skipped.append(season)
            print(f"{season}: already cached, skipping")
            continue
        try:
            C.save(fetch_season(season, args.limit), SPORT, season)
            done.append(season)
        except Exception as exc:                   # noqa: BLE001
            failed.append(season)
            print(f"{season}: FAILED - {str(exc)[:160]}")

    print()
    print("=" * 60)
    print(f"fetched : {done or 'nothing'}")
    print(f"skipped : {skipped or 'nothing'}")
    print(f"failed  : {failed or 'nothing'}")
    print(f"cache now holds: {C.cached_seasons(SPORT)}")
    if failed:
        print("\nCommit what succeeded and re-run for the rest. Nothing "
              "already on disk will be fetched again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
