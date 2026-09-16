"""Three blocking questions the depth probe raised but could not answer.

The depth probe said 2016-2025 has ample history. Good. It also showed 2026
week 3 returning zero games while the 2026 roster returned 31,070 players.
That combination is the whole reason this file exists, because there are
several explanations and they have very different consequences:

  - the season's weeks are numbered differently than assumed (harmless)
  - stats lag the games by some days (annoying, workable)
  - current-season stats need a paid tier (serious - kills in-season form)

Guessing which is a coin flip. Fetching is not.

The three questions:

  1. CURRENT SEASON. Which 2026 weeks actually return player stats, and do
     the games themselves show as completed? A week with completed games and
     no stats is an ingest problem. A week with no completed games is just
     the calendar, and means nothing is wrong.

  2. THE POSITION JOIN. games/players gives no position; roster does. If
     athlete ids from the stats endpoint match ids from the roster endpoint,
     the join is exact and the odd roster counts across seasons don't matter.
     If they don't match, positions must come from name matching, and the
     roster including non-FBS divisions suddenly matters a great deal. This
     measures the join rate rather than hoping for it.

  3. WHO ACTUALLY SCORES. Of the rows that produce fantasy points, what
     fraction get a position, and how are they distributed? A 99% join rate
     that misses every quarterback is not a 99% join rate.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter

import pandas as pd
import requests

VERSION = "v1"

log = logging.getLogger("cfb_ready")

CFBD = "https://api.collegefootballdata.com"
TIMEOUT = 45
SKILL = {"QB", "RB", "WR", "TE", "FB"}


def _key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set. Add it as a repository secret.")
    return key


def get(path: str, key: str, **params):
    url = f"{CFBD}/{path}"
    try:
        r = requests.get(url, params=params,
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=TIMEOUT)
    except Exception as exc:                       # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:140]}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:140]}"
    try:
        return r.json(), None
    except Exception as exc:                       # noqa: BLE001
        return None, f"not JSON ({type(exc).__name__}): {r.text[:140]}"


def athlete_ids(payload: list[dict]) -> set:
    """Every athlete id appearing anywhere in a games/players payload."""
    out = set()
    for game in payload or []:
        for team in game.get("teams") or []:
            for cat in team.get("categories") or []:
                for typ in cat.get("types") or []:
                    for ath in typ.get("athletes") or []:
                        if ath.get("id") is not None:
                            out.add(str(ath["id"]))
    return out


def scoring_ids(payload: list[dict]) -> set:
    """Athlete ids that recorded a stat we actually pay points for.

    Question 3 is about these, not about everyone who appeared. A long
    snapper joins or doesn't join and it changes nothing.
    """
    want = {("passing", "YDS"), ("passing", "TD"),
            ("rushing", "YDS"), ("rushing", "TD"),
            ("receiving", "REC"), ("receiving", "YDS"), ("receiving", "TD")}
    out = set()
    for game in payload or []:
        for team in game.get("teams") or []:
            for cat in team.get("categories") or []:
                cname = cat.get("name")
                for typ in cat.get("types") or []:
                    if (cname, typ.get("name")) not in want:
                        continue
                    for ath in typ.get("athletes") or []:
                        if ath.get("id") is not None:
                            out.add(str(ath["id"]))
    return out


def roster_index(key: str, season: int) -> tuple[dict, str | None]:
    """{athlete_id: position} for a season, plus whichever id field exists.

    CFBD has renamed roster fields before, so the id field is discovered
    rather than assumed - and the discovery is printed, because a silently
    wrong field name would look exactly like a zero join rate.
    """
    payload, err = get("roster", key, year=season)
    if err:
        return {}, err
    if not payload:
        return {}, "roster returned empty"
    sample = payload[0]
    id_field = next((f for f in ("id", "athlete_id", "athleteId", "player_id")
                     if f in sample), None)
    if id_field is None:
        return {}, f"no id field in roster; keys are {sorted(sample)[:12]}"
    log.info("season %d roster id field is %r", season, id_field)
    idx = {}
    for p in payload:
        pid = p.get(id_field)
        if pid is not None:
            idx[str(pid)] = str(p.get("position") or "").upper()
    return idx, None


# --- question 1 -------------------------------------------------------------

def current_season(key: str, season: int, last_week: int) -> pd.DataFrame:
    """Per week: games scheduled, games completed, and athlete-games of stats.

    'Completed' comes from the games endpoint, which is the control. Without
    it a zero is uninterpretable - it could mean the games have not happened.
    """
    rows = []
    for week in range(0, last_week + 1):
        games, gerr = get("games", key, year=season, week=week,
                          seasonType="regular")
        time.sleep(0.4)
        scheduled = len(games or [])
        completed = sum(1 for g in games or []
                        if g.get("completed") is True
                        or (g.get("home_points") is not None
                            and g.get("away_points") is not None))
        stats, serr = get("games/players", key, year=season, week=week,
                          seasonType="regular")
        time.sleep(0.4)
        n_ath = len(athlete_ids(stats)) if not serr else 0
        rows.append({
            "week": week, "scheduled": scheduled, "completed": completed,
            "stat_games": len(stats or []) if not serr else 0,
            "athletes": n_ath,
            "note": (gerr or "")[:40] or (serr or "")[:40] or "",
        })
        log.info("%d wk%d: %d scheduled, %d completed, %d athletes",
                 season, week, scheduled, completed, n_ath)
    return pd.DataFrame(rows)


def diagnose_current(df: pd.DataFrame) -> list[str]:
    """Name the cause, from the evidence, rather than describing the symptom."""
    lines = ["", "-" * 72, "Q1  CURRENT SEASON", "-" * 72]
    if df.empty:
        return lines + ["no data"]
    played = df[df["completed"] > 0]
    if played.empty:
        return lines + ["No completed games in any probed week. Either the",
                        "season has not started or the games endpoint is",
                        "unavailable. Not a stats problem."]
    dry = played[played["athletes"] == 0]
    wet = played[played["athletes"] > 0]
    if dry.empty:
        lines += [f"Every week with completed games has player stats "
                  f"(weeks {int(wet['week'].min())}-{int(wet['week'].max())}).",
                  "Current-season stats are available. In-season form",
                  "features can be built. Nothing blocks the build."]
        return lines
    lines += [f"Weeks with completed games but NO stats: "
              + ", ".join(str(int(w)) for w in dry["week"])]
    if wet.empty:
        lines += ["", "No completed week returns stats at all. This is a",
                  "hard block: the API gives history but not the current",
                  "season. Most likely a tier restriction.",
                  "CONSEQUENCE: the model can be fitted but cannot see this",
                  "season's usage. Do not proceed as if it can."]
    else:
        newest_wet = int(wet["week"].max())
        lines += ["", f"Stats exist through week {newest_wet} and stop.",
                  "That is a LAG, not a restriction - the most recent week",
                  "has not been ingested yet.",
                  "CONSEQUENCE: workable. Features must use weeks strictly",
                  "before the newest available one, and the build has to",
                  "treat 'newest week with stats' as a fetched fact rather",
                  "than assuming it is last week."]
    return lines


# --- questions 2 and 3 ------------------------------------------------------

def join_rate(key: str, season: int, week: int) -> list[str]:
    lines = ["", "-" * 72, f"Q2/Q3  POSITION JOIN  ({season} week {week})",
             "-" * 72]
    stats, serr = get("games/players", key, year=season, week=week,
                      seasonType="regular")
    if serr:
        return lines + [f"stats unavailable: {serr}"]
    idx, rerr = roster_index(key, season)
    if rerr:
        return lines + [f"roster unavailable: {rerr}"]

    every = athlete_ids(stats)
    scorers = scoring_ids(stats)
    if not every:
        return lines + ["no athletes in payload"]

    hit_all = sum(1 for a in every if a in idx)
    hit_sc = sum(1 for a in scorers if a in idx)
    pos = Counter(idx.get(a, "(unmatched)") for a in scorers)

    lines += [
        f"roster entries         : {len(idx):,}",
        f"athletes with stats    : {len(every):,}",
        f"  matched to a position: {hit_all:,}  ({100*hit_all/len(every):.1f}%)",
        f"athletes who SCORED    : {len(scorers):,}",
        f"  matched to a position: {hit_sc:,}  "
        f"({100*hit_sc/max(1,len(scorers)):.1f}%)",
        "",
        "Positions among scorers:",
    ]
    for p, n in pos.most_common(14):
        mark = "  <-- SKILL" if p in SKILL else ""
        lines.append(f"  {p or '(blank)':<14}{n:>6}{mark}")

    rate = hit_sc / max(1, len(scorers))
    lines.append("")
    if rate >= 0.97:
        lines += ["Ids join cleanly. Positions come from the roster endpoint",
                  "by id, no name matching needed, and the varying roster",
                  "sizes across seasons are irrelevant."]
    elif rate >= 0.80:
        lines += ["Partial join. Usable, but the unmatched scorers need a",
                  "name-based fallback and that fallback needs measuring,",
                  "not assuming."]
    else:
        lines += ["Ids do NOT join. Positions must come from name matching",
                  "against a roster that may include non-FBS divisions.",
                  "That is a real piece of work and it belongs before the",
                  "model, not after."]
    return lines


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--last-week", type=int, default=6)
    p.add_argument("--join-season", type=int, default=2025,
                   help="a complete season, for the position join test")
    p.add_argument("--join-week", type=int, default=3)
    args = p.parse_args()

    print(f"cfb_ready {VERSION}")
    print("=" * 72)
    key = _key()

    cur = current_season(key, args.season, args.last_week)
    print()
    print(cur.to_string(index=False))
    print("\n".join(diagnose_current(cur)))

    print("\n".join(join_rate(key, args.join_season, args.join_week)))

    # And the same join test on the CURRENT season, because a roster that
    # exists is not the same as a roster whose ids match this year's stats.
    if args.season != args.join_season:
        live = cur[cur["athletes"] > 0]
        if not live.empty:
            wk = int(live["week"].max())
            print("\n".join(join_rate(key, args.season, wk)))
        else:
            print(f"\n(skipping current-season join test: no {args.season} "
                  f"week returned stats)")

    cur.to_csv("cfb_ready.csv", index=False)
    print("\nwrote cfb_ready.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
