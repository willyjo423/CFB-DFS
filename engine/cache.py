"""Fetch each season once, ever. Any sport.

Two different sports have now hit the same wall from opposite directions.
College football re-downloaded four seasons on every verification run and
exhausted a monthly API quota mid-grade. Baseball has no quota at all, but a
season is 2,430 box scores - about twelve minutes of HTTP - and paying that
on every run is twelve minutes nobody gets back.

A finished season never changes. Downloading it more than once is not a
tradeoff; it is waste that eventually costs the ability to work at all.

So history lives in gzipped CSV under data/, one file per sport-season,
fetched once and committed. Runs read from disk. The only season worth
re-fetching is the current one, because it grows, and `ensure(refresh=...)`
exists for that and only that.

Gzipped CSV rather than parquet because parquet needs pyarrow, which cannot
be installed in the environment this was written in - so a parquet cache
could only ever be tested somewhere other than where it was written. Several
bugs in this project hid in exactly that gap. The cost is that CSV forgets
types, and one of those matters: a player id is digits but is not a number,
and read back as an integer it silently stops matching anything. So the
dtypes are stated on read rather than inferred.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE = Path("data")

# Columns that look numeric and are not. A player id read back as int64 stops
# matching the board, silently, and the join simply gets worse.
TEXT_COLUMNS = {"player_id": "string", "athlete_id": "string",
                "name": "string", "team": "string", "school": "string",
                "opponent": "string", "position": "string",
                "game_id": "string", "game_pk": "string", "date": "string"}


def path_for(sport: str, season: int) -> Path:
    return CACHE / f"{sport}_{season}.csv.gz"


def have(sport: str, season: int) -> bool:
    return path_for(sport, season).exists()


def cached_seasons(sport: str) -> list[int]:
    if not CACHE.exists():
        return []
    out = []
    for p in sorted(CACHE.glob(f"{sport}_*.csv.gz")):
        try:
            out.append(int(p.name.split("_")[-1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return out


def save(df: pd.DataFrame, sport: str, season: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = path_for(sport, season)
    # Columns holding python sets or lists cannot round-trip through CSV and
    # are all cheaply derived from `name`, so they are rebuilt on load rather
    # than stored - cheaper than serialising and impossible to get stale.
    drop = [c for c in df.columns
            if df[c].map(lambda v: isinstance(v, (set, list, dict))).any()]
    out = df.drop(columns=drop) if drop else df
    out.to_csv(p, index=False, compression="gzip")
    log.info("wrote %s (%d rows, %.1f MB)", p, len(out),
             p.stat().st_size / 1e6)
    return p


def load(sport: str, seasons: list[int]) -> pd.DataFrame:
    frames, missing = [], []
    for s in seasons:
        p = path_for(sport, s)
        if not p.exists():
            missing.append(s)
            continue
        dtypes = {k: v for k, v in TEXT_COLUMNS.items()}
        frames.append(pd.read_csv(p, compression="gzip", dtype=dtypes,
                                  low_memory=False))
    if missing:
        log.warning("%s: no cached file for %s - run the fetch workflow",
                    sport, missing)
    if not frames:
        raise FileNotFoundError(
            f"no cached {sport} history for {seasons}. Run the fetch "
            f"workflow first; grading must not re-download the archive on "
            f"every run.")
    df = pd.concat(frames, ignore_index=True)
    log.info("%s: loaded %d rows from cache", sport, len(df))
    return df


def ensure(sport: str, seasons: list[int], fetch_one,
           refresh: list[int] | None = None) -> pd.DataFrame:
    """Load from cache, fetching ONLY what is missing or explicitly refreshed.

    `fetch_one(season) -> DataFrame` is the sport's own downloader. The cache
    never knows how a sport gets its data, only that it should not ask twice.
    """
    refresh = set(refresh or [])
    for s in seasons:
        if have(sport, s) and s not in refresh:
            continue
        log.info("%s %d: fetching (%s)", sport, s,
                 "refresh" if have(sport, s) else "missing")
        save(fetch_one(s), sport, s)
    return load(sport, seasons)
