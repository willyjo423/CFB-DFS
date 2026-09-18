"""Statcast, reduced to one row per player per game.

    python mlb_statcast.py --seasons 2025 2026
    python mlb_statcast.py --seasons 2026 --refresh 2026

Why not the leaderboards
------------------------
Baseball Savant publishes ready-made expected-statistics leaderboards, and
they are the wrong shape for this project in a way that would not announce
itself. A leaderboard is a SEASON AGGREGATE: Aaron Judge's 2026 xwOBA
includes September when you are projecting May. Joined onto a historical row
it hands the model the future, every backtest improves, and the improvement is
entirely fictional. This project has already shipped one feature that graded
worse than nothing; a leaked one would grade BETTER than the truth, which is
far harder to catch.

So this reads the pitch-level feed instead, where every row carries its own
date, and reduces it to per-player-per-game rates. Those go through the same
shifted EWM as every other feature, and a game cannot see itself.

What is kept, and why these
---------------------------
The point of Statcast is that it measures the SWING rather than the RESULT.
A hitter's actual batting line over two weeks is mostly luck - where the ball
happened to land - and stabilises after hundreds of plate appearances. What
he did to the ball stabilises in dozens.

  xwobacon      expected wOBA on contact. What his batted balls were worth
                given how hard and at what angle they left the bat, rather
                than whether they found a fielder.
  barrel_rate   the share of batted balls in the launch-speed-and-angle
                bucket that produces extra-base hits far more often than
                anything else. This is the home-run signal.
  hard_hit_rate share of batted balls at 95mph or more. Noisier than barrels
                individually but far more of them, so it steadies sooner.
  whiff_rate    swings that missed. For a pitcher this is the strikeout
                signal, measured on pitches rather than on outcomes, which is
                what the last attempt at a matchup feature lacked.

Both sides of every pitch are aggregated, so a batter row and a pitcher row
come out of the same fetch. A pitcher's `xwobacon` is what hitters did to
him; a hitter's is what he did.

The download
------------
A season is roughly 700,000 pitches. Savant will not hand that over in one
request, and should not be asked to, so it is fetched a week at a time and
reduced immediately - the raw pitches are never all in memory at once and
never reach disk. What is cached is the aggregate, a few megabytes a season.

If the feed is unreachable this returns empty and says so loudly rather than
raising. A missing Statcast file must degrade the model to what it was
before, not take the build down.
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

from engine import cache as C

log = logging.getLogger("mlb_statcast")

SPORT = "mlb_statcast"
BASE = "https://baseballsavant.mlb.com/statcast_search/csv"

# A week at a time. Longer ranges time out on Savant's side; shorter ones
# multiply the request count for no gain.
CHUNK_DAYS = 7

# Baseball is played between these, near enough. Fetching January costs a
# request and returns nothing.
SEASON_START = (3, 1)
SEASON_END = (11, 15)

# Savant is free and unmetered and that is a reason to be polite, not a
# reason not to be.
PAUSE_SECONDS = 1.0
RETRIES = 3
TIMEOUT = 180

# 95 mph. The line the league itself uses for "hard hit", so the feature
# means the same thing here as everywhere else it is quoted.
HARD_HIT_MPH = 95.0

# Savant scores every batted ball 1-6 on launch speed and angle together.
# Six is a barrel: the combination that produces a slugging percentage north
# of 1.500 league-wide. It is a single column and it is already computed, so
# there is no reason to re-derive it from speed and angle and get it subtly
# wrong.
BARREL_CODE = 6


def _chunks(season: int):
    start = date(season, *SEASON_START)
    end = min(date(season, *SEASON_END), date.today())
    while start <= end:
        stop = min(start + timedelta(days=CHUNK_DAYS - 1), end)
        yield start, stop
        start = stop + timedelta(days=1)


def _fetch_range(start: date, stop: date) -> pd.DataFrame:
    """Every pitch between two dates, or an empty frame."""
    url = (f"{BASE}?all=true&type=details"
           f"&game_date_gt={start.isoformat()}"
           f"&game_date_lt={stop.isoformat()}"
           f"&hfSea={start.year}%7C&player_type=batter")
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "dfs-model/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = r.read()
            if not raw.strip():
                return pd.DataFrame()
            return pd.read_csv(io.BytesIO(raw), low_memory=False)
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as exc:
            last = exc
            wait = 2 ** attempt
            log.warning("%s..%s failed (%s: %s), retrying in %ds",
                        start, stop, type(exc).__name__, str(exc)[:70], wait)
            time.sleep(wait)
        except Exception as exc:                               # noqa: BLE001
            last = exc
            log.warning("%s..%s did not parse (%s: %s)", start, stop,
                        type(exc).__name__, str(exc)[:70])
            break
    log.error("gave up on %s..%s (%s)", start, stop,
              type(last).__name__ if last else "unknown")
    return pd.DataFrame()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def reduce_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    """Pitch rows -> one row per (game_pk, player_id, side).

    Kept as COUNTS rather than rates, because counts add across chunks and
    rates do not. A week's rate cannot be averaged with another week's rate
    without weighting it, and doing that arithmetic in two places is how the
    two stop agreeing. The rates are computed once, at the end.
    """
    if raw.empty:
        return pd.DataFrame()

    need = {"game_pk", "batter", "pitcher", "game_date"}
    if not need.issubset(raw.columns):
        log.error("the Statcast feed is missing %s - its columns have "
                  "changed and this parser needs updating",
                  sorted(need - set(raw.columns)))
        return pd.DataFrame()

    df = pd.DataFrame({
        "game_pk": raw["game_pk"].astype("Int64").astype("string"),
        "date": pd.to_datetime(raw["game_date"], errors="coerce"),
        "batter": _num(raw, "batter").astype("Int64").astype("string"),
        "pitcher": _num(raw, "pitcher").astype("Int64").astype("string"),
        "xwoba": _num(raw, "estimated_woba_using_speedangle"),
        "launch_speed": _num(raw, "launch_speed"),
        "lsa": _num(raw, "launch_speed_angle"),
        "description": raw.get("description", pd.Series(
            "", index=raw.index)).astype(str),
    })

    # A batted ball is a pitch that produced a measured exit velocity.
    df["bip"] = df["launch_speed"].notna().astype(float)
    df["barrel"] = (df["lsa"] == BARREL_CODE).astype(float)
    df["hard_hit"] = (df["launch_speed"] >= HARD_HIT_MPH).astype(float)
    df["xwoba_sum"] = df["xwoba"].fillna(0.0) * df["xwoba"].notna()
    df["xwoba_n"] = df["xwoba"].notna().astype(float)

    d = df["description"]
    df["swing"] = d.isin(["hit_into_play", "foul", "swinging_strike",
                          "swinging_strike_blocked", "foul_tip", "foul_bunt",
                          "missed_bunt", "bunt_foul_tip"]).astype(float)
    df["whiff"] = d.isin(["swinging_strike", "swinging_strike_blocked",
                          "missed_bunt"]).astype(float)
    df["pitches"] = 1.0

    cols = ["bip", "barrel", "hard_hit", "xwoba_sum", "xwoba_n", "swing",
            "whiff", "pitches"]
    out = []
    for side, key in (("batter", "batter"), ("pitcher", "pitcher")):
        g = (df.groupby(["game_pk", key, "date"], dropna=True)[cols]
               .sum().reset_index()
               .rename(columns={key: "player_id"}))
        g["side"] = side
        out.append(g)
    return pd.concat(out, ignore_index=True)


def fetch_one(season: int) -> pd.DataFrame:
    """One season of Statcast, already reduced to player-games."""
    parts, weeks = [], 0
    for start, stop in _chunks(season):
        raw = _fetch_range(start, stop)
        weeks += 1
        if raw.empty:
            continue
        part = reduce_pitches(raw)
        if not part.empty:
            parts.append(part)
            log.info("%s..%s: %d pitches -> %d player-games",
                     start, stop, len(raw), len(part))
        time.sleep(PAUSE_SECONDS)

    if not parts:
        log.error("season %d: NOTHING was fetched from Statcast across %d "
                  "weeks. The model will run without it.", season, weeks)
        return pd.DataFrame(columns=["game_pk", "player_id", "date", "side"])

    df = pd.concat(parts, ignore_index=True)
    # A player-game can straddle two chunks only if a chunk boundary splits a
    # game, which it cannot - but summing again costs nothing and makes the
    # invariant true by construction rather than by argument.
    keys = ["game_pk", "player_id", "date", "side"]
    num = [c for c in df.columns if c not in keys]
    df = df.groupby(keys, dropna=True)[num].sum().reset_index()

    bip = df["bip"].replace(0, np.nan)
    df["sc_xwobacon"] = (df["xwoba_sum"] / df["xwoba_n"].replace(0, np.nan))
    df["sc_barrel_rate"] = df["barrel"] / bip
    df["sc_hard_hit_rate"] = df["hard_hit"] / bip
    df["sc_whiff_rate"] = df["whiff"] / df["swing"].replace(0, np.nan)
    df["season"] = season

    log.info("season %d: %d player-games (%d batter, %d pitcher)", season,
             len(df), int((df["side"] == "batter").sum()),
             int((df["side"] == "pitcher").sum()))
    return df


RATE_COLUMNS = ["sc_xwobacon", "sc_barrel_rate", "sc_hard_hit_rate",
                "sc_whiff_rate"]


def load(seasons: list[int]) -> pd.DataFrame:
    """The cached aggregate, or empty if it has never been fetched.

    Empty is a supported answer. The Statcast features are optional by
    design: a model that cannot build without them would make the fetch a
    single point of failure for a board that has to go out before first
    pitch.
    """
    try:
        df = C.load(SPORT, seasons)
    except FileNotFoundError:
        log.warning("no Statcast cache for %s. The hitter and pitcher models "
                    "will run without it - run mlb_statcast.py to build it.",
                    seasons)
        return pd.DataFrame()
    for c in ("game_pk", "player_id", "side"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--refresh", type=int, nargs="*", default=None,
                   help="seasons to re-download even if cached")
    args = p.parse_args(argv)

    df = C.ensure(SPORT, args.seasons, fetch_one, refresh=args.refresh)
    if df.empty:
        print("NOTHING was cached. Statcast is unreachable or its columns "
              "have changed.")
        return 1

    print(f"\n{len(df):,} player-games cached for {sorted(set(df['season']))}")
    for side in ("batter", "pitcher"):
        s = df[df["side"] == side]
        if s.empty:
            continue
        print(f"\n  {side}s: {len(s):,} rows")
        for c in RATE_COLUMNS:
            v = pd.to_numeric(s[c], errors="coerce")
            print(f"    {c:<18} present on {100 * v.notna().mean():5.1f}% "
                  f"of rows, median {v.median():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
