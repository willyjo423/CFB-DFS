"""How much college football history actually exists, and is it usable.

The NFL model trains from 2017 because nflverse publishes clean, positioned,
season-long player data back further than that. Nothing guarantees CFBD does
the same, and a quantile model fitted to two weeks of one season is not a
model - it is a rounding error with a confidence interval.

So this answers three questions before a single line of the CFB model gets
written, and it answers them by fetching, not by assuming:

  1. How far back does games/players return real player stats?
  2. Does a POSITION exist for those players in that season? The roster
     endpoint is the only source of it, and a player-game row without a
     position cannot fill a roster slot, cannot get a position baseline, and
     cannot be graded by position. Stats without positions are unusable here.
  3. How many player-games per season, so the training volume is a number
     rather than a hope. model.py refuses to fit below 400 played rows; the
     NFL fit used tens of thousands.

Run it from GitHub Actions - the CFBD API is not reachable from the machine
this was written on, which is exactly why it is a workflow and not a claim.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import pandas as pd
import requests

VERSION = "v1"

log = logging.getLogger("cfb_depth")

CFBD = "https://api.collegefootballdata.com"
TIMEOUT = 45

# One representative week per season. Week 3 is deliberate: week 1 is full of
# FCS cupcakes that inflate volume and distort scoring, and by week 3 the
# schedule is mostly real. Probing every week of every season would be a
# thousand calls and would tell us nothing week 3 does not.
PROBE_WEEK = 3

# What a fantasy-relevant row needs. Checked by name against what comes back,
# because a category that silently disappeared in 2016 would otherwise show up
# as "the model got worse in older seasons" three weeks from now.
WANTED = {
    ("passing", "YDS"), ("passing", "TD"), ("passing", "INT"),
    ("rushing", "YDS"), ("rushing", "TD"),
    ("receiving", "REC"), ("receiving", "YDS"), ("receiving", "TD"),
    ("fumbles", "LOST"),
}

SKILL = {"QB", "RB", "WR", "TE", "FB"}


def _key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set. Add it as a repository secret.")
    return key


def get(path: str, key: str, **params):
    """One call, with the failure mode printed rather than swallowed."""
    url = f"{CFBD}/{path}"
    try:
        r = requests.get(url, params=params,
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=TIMEOUT)
    except Exception as exc:                       # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:120]}"
    try:
        return r.json(), None
    except Exception as exc:                       # noqa: BLE001
        return None, f"not JSON ({type(exc).__name__}): {r.text[:120]}"


def categories(payload: list[dict]) -> set[tuple[str, str]]:
    """Which (category, type) pairs this payload actually contains."""
    seen = set()
    for game in payload or []:
        for team in game.get("teams") or []:
            for cat in team.get("categories") or []:
                cname = cat.get("name")
                for typ in cat.get("types") or []:
                    seen.add((cname, typ.get("name")))
    return seen


def player_games(payload: list[dict]) -> tuple[int, int]:
    """(rows, distinct athletes) without pivoting - this is a count, not a load."""
    athletes = set()
    rows = 0
    for game in payload or []:
        for team in game.get("teams") or []:
            for cat in team.get("categories") or []:
                for typ in cat.get("types") or []:
                    for ath in typ.get("athletes") or []:
                        rows += 1
                        athletes.add((game.get("id"), ath.get("id")))
    return rows, len(athletes)


def roster_positions(key: str, season: int) -> tuple[int, int, str | None]:
    """(players, skill players, error) on the roster endpoint for a season.

    The stats endpoint gives no position. If this comes back empty for a
    season, that season's stats are unusable no matter how complete they are,
    because nothing can be assigned to a roster slot.
    """
    payload, err = get("roster", key, year=season)
    if err:
        return 0, 0, err
    total = len(payload or [])
    skill = sum(1 for p in payload or []
                if str(p.get("position") or "").upper() in SKILL)
    return total, skill, None


def probe(seasons: list[int], key: str) -> pd.DataFrame:
    out = []
    for season in seasons:
        payload, err = get("games/players", key, year=season,
                           week=PROBE_WEEK, seasonType="regular")
        time.sleep(0.5)
        if err:
            out.append({"season": season, "games": 0, "rows": 0,
                        "athletes": 0, "missing_cats": "-", "roster": 0,
                        "skill": 0, "note": err})
            log.warning("%d: %s", season, err)
            continue
        games = len(payload or [])
        rows, athletes = player_games(payload)
        have = categories(payload)
        missing = WANTED - have
        n_roster, n_skill, rerr = roster_positions(key, season)
        time.sleep(0.5)
        out.append({
            "season": season, "games": games, "rows": rows,
            "athletes": athletes,
            "missing_cats": ",".join(f"{c}/{t}" for c, t in sorted(missing))
                            or "none",
            "roster": n_roster, "skill": n_skill,
            "note": rerr or "",
        })
        log.info("%d: %d games, %d athlete-games, roster %d (%d skill)",
                 season, games, athletes, n_roster, n_skill)
    return pd.DataFrame(out)


def verdict(df: pd.DataFrame) -> list[str]:
    """Say plainly which seasons are trainable, and what the volume buys.

    A season counts as usable only if it has BOTH stats and positions. Either
    one alone is useless, which is the whole reason both are probed.
    """
    lines = ["", "=" * 72, "VERDICT", "=" * 72]
    usable = df[(df["athletes"] > 0) & (df["skill"] > 0)
                & (df["missing_cats"] == "none")]
    if usable.empty:
        lines += ["No season has stats AND positions AND all scoring",
                  "categories. The CFB model cannot be fitted from this API",
                  "as probed. Do not proceed to modelling - fix ingest first."]
        return lines

    first, last = int(usable["season"].min()), int(usable["season"].max())
    per_week = float(usable["athletes"].mean())
    # Regular season is ~15 weeks; week 3 is representative of a full one.
    per_season = per_week * 15
    total = per_season * len(usable)
    lines += [
        f"Usable seasons : {first}-{last}  ({len(usable)} of {len(df)} probed)",
        f"Athlete-games  : ~{per_week:,.0f} in week {PROBE_WEEK}",
        f"                 ~{per_season:,.0f} per season (15 weeks)",
        f"                 ~{total:,.0f} total across usable seasons",
        "",
        "All positions, not just skill. Filtering to QB/RB/WR/TE will cut",
        "this substantially - the fitted population is what matters, and it",
        "is smaller than the raw count. Grade on the filtered number.",
    ]
    if total < 20_000:
        lines += ["", "THIN. Below what the NFL fit used. Expect wider",
                  "quantiles and weaker availability separation."]
    else:
        lines += ["", "Sufficient depth to fit. Set TRAIN_START_SEASON to",
                  f"{first} and let the walk-forward decide if older seasons",
                  "actually help - cheap to test, expensive to assume."]

    broken = df[(df["athletes"] > 0) & (df["missing_cats"] != "none")]
    if not broken.empty:
        lines += ["", "Seasons with stats but MISSING scoring categories:"]
        for r in broken.itertuples(index=False):
            lines.append(f"  {r.season}: {r.missing_cats}")
        lines += ["These would score wrong, not merely score less. Excluded."]

    noposition = df[(df["athletes"] > 0) & (df["skill"] == 0)]
    if not noposition.empty:
        lines += ["", "Seasons with stats but NO roster positions: "
                  + ", ".join(str(int(s)) for s in noposition["season"])]
        lines += ["Unusable - a player-game with no position cannot fill a",
                  "roster slot or get a position baseline."]
    return lines


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=2014)
    p.add_argument("--last", type=int, default=2026)
    args = p.parse_args()

    print(f"cfb_depth {VERSION}   probing week {PROBE_WEEK} "
          f"of {args.first}-{args.last}")
    print("=" * 72)

    key = _key()
    seasons = list(range(args.first, args.last + 1))
    df = probe(seasons, key)

    print()
    print(df.to_string(index=False))
    print("\n".join(verdict(df)))

    df.to_csv("cfb_depth.csv", index=False)
    print("\nwrote cfb_depth.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
