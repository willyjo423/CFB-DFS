"""Baseball: the board, the box scores, and the join.

Why this should be far quicker than college football was
--------------------------------------------------------
The CFB build lost most of its time to identity: DraftKings' team codes did
not match CFBD's schools, names disagreed in four separate ways, 622 schools
were in scope, and the API had a monthly quota that ran out mid-build.

None of that applies here. The MLB StatsAPI is free, needs no key, has no
quota, covers thirty fixed teams, and publishes a STABLE NUMERIC PLAYER ID on
every box score. DraftKings publishes the same player's id in its own feed.
So the join is on an integer, not on a name, and the entire Matt/Matthew
Fuller problem simply does not arise.

What is genuinely harder than football
--------------------------------------
**Two sports in one.** Hitters and pitchers score under different rules, fill
different roster slots, and need different features. They get two specs and
two fitted models, not one model with a position flag - a strikeout means
opposite things to the two of them.

**Innings pitched are not decimal.** "6.2" means six and two-thirds, not six
point two. Reading it as a float understates every start by about a third of
an inning and silently mis-scores every pitcher on the board.

**Lineups land two hours before first pitch.** Batting order is the single
biggest driver of a hitter's day and is not knowable earlier. That is a live
problem for projections, not a historical one, and it is why the board is
fetched close to lock rather than the night before.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)"}
TIMEOUT = 45
RETRIES = 3

STATS = "https://statsapi.mlb.com/api/v1"
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=MLB"
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")

# DraftKings MLB Classic. Two scoring systems, because they are two games.
#
# These values are asserted, not derived, which makes them the most likely
# thing in this file to be wrong - so the verifier recomputes DraftKings'
# own published points per game from these rules and compares. That check
# caught nothing in college football because the rules were right; it is
# here precisely so that being wrong is loud rather than silent.
HITTER_SCORING = {
    "single": 3.0, "double": 5.0, "triple": 8.0, "home_run": 10.0,
    "rbi": 2.0, "run": 2.0, "walk": 2.0, "hbp": 2.0, "stolen_base": 5.0,
}
PITCHER_SCORING = {
    "innings": 2.25, "strikeout": 2.0, "win": 4.0,
    "earned_run": -2.0, "hit_allowed": -0.6, "walk_allowed": -0.6,
    "hbp_allowed": -0.6, "complete_game": 2.5, "shutout": 2.5,
    "no_hitter": 5.0,
}

HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF"]
PITCHER_POSITIONS = ["SP", "RP", "P"]


class Unavailable(RuntimeError):
    """A source this build cannot proceed without did not answer."""


# ----------------------------------------------------------------- transport

def _get(url: str, params=None, headers=None):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers or UA,
                             timeout=TIMEOUT)
        except Exception as exc:                   # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:110]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as exc:               # noqa: BLE001
                raise Unavailable(
                    f"{url[:70]} returned 200 but not JSON "
                    f"({type(exc).__name__}): {r.content[:100]!r}") from exc
        last = f"HTTP {r.status_code}: {r.text[:110]}"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        break
    raise Unavailable(f"{url[:70]} -> {last}")


# --------------------------------------------------------------- statsapi

def innings(value) -> float:
    """Innings pitched, which are thirds and not decimals.

    "6.2" is six and TWO THIRDS. Read as a float it is 6.2, which understates
    the start by about a sixth of an inning and, at 2.25 points an inning,
    mis-scores every pitcher on the board in the same direction. The error is
    small enough per start to look like rounding and large enough across a
    slate to change which pitcher the optimiser picks.
    """
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    if "." not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0
    whole, _, frac = s.partition(".")
    try:
        w = float(whole or 0)
    except ValueError:
        return 0.0
    thirds = {"0": 0.0, "1": 1.0 / 3.0, "2": 2.0 / 3.0}
    return w + thirds.get(frac[:1], 0.0)


def schedule(season: int, start: str | None = None, end: str | None = None
             ) -> pd.DataFrame:
    """Every regular-season game, with its id and whether it finished."""
    params = {"sportId": 1, "season": season, "gameType": "R"}
    if start and end:
        params.update({"startDate": start, "endDate": end})
    payload = _get(f"{STATS}/schedule", params=params)
    rows = []
    for day in payload.get("dates") or []:
        for g in day.get("games") or []:
            state = (g.get("status") or {}).get("codedGameState")
            teams = g.get("teams") or {}
            rows.append({
                "game_pk": g.get("gamePk"),
                "date": day.get("date"),
                "final": state == "F",
                "home_id": ((teams.get("home") or {}).get("team") or {}
                            ).get("id"),
                "away_id": ((teams.get("away") or {}).get("team") or {}
                            ).get("id"),
                "home": ((teams.get("home") or {}).get("team") or {}
                         ).get("name"),
                "away": ((teams.get("away") or {}).get("team") or {}
                         ).get("name"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise Unavailable(f"{season}: the schedule returned no games")
    log.info("%d: %d regular-season games, %d final",
             season, len(df), int(df["final"].sum()))
    return df


def probable_pitchers(date: str) -> dict[str, str]:
    """Today's announced starters: player id -> the game he starts in.

    The single most expensive thing a baseball board can get wrong. Every
    pitcher on a 26-man roster is priced; two of them start. The one who is
    not starting is cheap, and points per dollar is precisely the statistic a
    zero-inning pitcher maximises - so every objective picks him, every time.
    A star appearing in all ten lineups on a day he is not pitching is not a
    strange result; it is the only result a board without this can give.

    The league announces probables days ahead and publishes them free on the
    same endpoint the box scores come from. There is no excuse for a page that
    does not read them.

    An empty result means the league lists none, which the caller must treat
    as "do not publish" rather than as "nobody is starting today".
    """
    payload = _get(f"{STATS}/schedule",
                   params={"sportId": 1, "date": date, "gameType": "R",
                           "hydrate": "lineups,probablePitcher"})
    ids, names = {}, {}
    order_id, order_name, posted_teams = {}, {}, set()
    games = sides = posted = 0

    for day in payload.get("dates") or []:
        for g in day.get("games") or []:
            games += 1
            teams = g.get("teams") or {}
            lineups = g.get("lineups") or {}

            for side in ("home", "away"):
                info = (teams.get(side) or {}).get("team") or {}
                team = info.get("abbreviation") or info.get("name") or "?"

                p = (teams.get(side) or {}).get("probablePitcher") or {}
                pid, full = p.get("id"), p.get("fullName")
                if pid is not None or full:
                    sides += 1
                    if pid is not None:
                        ids[str(pid)] = team
                    if full:
                        names[normalise(full)] = team

                # The batting order, once it is posted - which is about two
                # hours before first pitch. Nine names, IN ORDER, and the
                # order is the point: leading off is roughly one extra plate
                # appearance a game over batting eighth, which is the single
                # largest thing separating one hitter's day from another's.
                nine = lineups.get(f"{side}Players") or []
                if nine:
                    posted += 1
                    posted_teams.add(str(team))
                    for slot, pl in enumerate(nine, start=1):
                        if pl.get("id") is not None:
                            order_id[str(pl["id"])] = slot
                        if pl.get("fullName"):
                            order_name[normalise(pl["fullName"])] = slot

    # Both keys are kept for both things, because neither survives alone. The
    # league's id is correct, and DraftKings' copy of it came back empty for a
    # whole 278-row board - the run where a filter keyed only on the id would
    # have dropped every pitcher on the slate rather than only the ones
    # sitting.
    log.info("%s: %d games, %d of %d starters announced, %d of %d lineups "
             "posted (%d batters in order)", date, games, sides, games * 2,
             posted, games * 2, len(order_id))
    if games and posted < games * 2:
        log.warning("%d of %d lineups are not posted yet - those teams' "
                    "hitters cannot be confirmed, and a hitter who is rested "
                    "scores zero. Lineups go up about two hours before first "
                    "pitch.", games * 2 - posted, games * 2)
    return {"ids": ids, "names": names,
            "order_id": order_id, "order_name": order_name,
            "posted_teams": posted_teams,
            "games": games, "announced": sides, "posted": posted}


def boxscore(game_pk: int) -> dict:
    return _get(f"{STATS}/game/{game_pk}/boxscore")


def describe_payload(box: dict) -> str:
    """What the box score actually contains, for the first run to print.

    Written because this file's scoring rules are asserted from memory and
    the field names are not. A run that prints the real keys turns "the
    numbers look a bit off" into "hitByPitch is called hitByPitches here".
    """
    lines = ["box score shape:"]
    teams = box.get("teams") or {}
    lines.append(f"  top-level keys : {sorted(box)[:10]}")
    lines.append(f"  team keys      : {sorted(teams)}")
    for side in ("away", "home"):
        players = (teams.get(side) or {}).get("players") or {}
        if not players:
            continue
        lines.append(f"  {side}: {len(players)} players")
        for _, p in list(players.items())[:2]:
            st = p.get("stats") or {}
            bat = st.get("batting") or {}
            pit = st.get("pitching") or {}
            lines.append(f"    {(p.get('person') or {}).get('fullName')} "
                         f"({(p.get('position') or {}).get('abbreviation')})")
            if bat:
                lines.append(f"      batting keys : {sorted(bat)}")
            if pit:
                lines.append(f"      pitching keys: {sorted(pit)}")
        break
    return "\n".join(lines)


def _decisions(box: dict) -> set:
    """Player ids credited with a win, if the payload carries a decisions
    block at all.

    The boxscore endpoint does NOT: its top-level keys are copyright, info,
    officials, pitchingNotes, teams and topPerformers. The first live run
    found zero wins across 120 games, which is four points missing from
    every winning pitcher.

    The win is in fact sitting in each pitcher's own stats block as `wins`,
    which is where it is read from now. This stays as a fallback for the
    live-feed payload shape, which does carry decisions.
    """
    out = set()
    w = (box.get("decisions") or {}).get("winner") or {}
    if w.get("id") is not None:
        out.add(str(w["id"]))
    return out


def player_games(box: dict, game: dict) -> list[dict]:
    """One row per player who appeared, hitting and pitching side by side."""
    rows = []
    teams = box.get("teams") or {}
    winners = _decisions(box)
    for side in ("away", "home"):
        blob = teams.get(side) or {}
        team = ((blob.get("team") or {}).get("name")
                or game.get(side))
        opp = game.get("home" if side == "away" else "away")
        for _, p in (blob.get("players") or {}).items():
            person = p.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            st = p.get("stats") or {}
            bat, pit = st.get("batting") or {}, st.get("pitching") or {}
            if not bat and not pit:
                continue
            pos = (p.get("position") or {}).get("abbreviation")
            hits = _n(bat.get("hits"))
            doubles = _n(bat.get("doubles"))
            triples = _n(bat.get("triples"))
            hr = _n(bat.get("homeRuns"))
            rows.append({
                "player_id": str(pid),
                "name": person.get("fullName"),
                "team": team, "opponent": opp,
                "is_home": 1 if side == "home" else 0,
                "position": pos,
                "game_pk": game.get("game_pk"),
                "date": game.get("date"),
                # hitting
                "single": max(0.0, hits - doubles - triples - hr),
                "double": doubles, "triple": triples, "home_run": hr,
                "rbi": _n(bat.get("rbi")), "run": _n(bat.get("runs")),
                "walk": _n(bat.get("baseOnBalls")),
                "hbp": _n(bat.get("hitByPitch")),
                "stolen_base": _n(bat.get("stolenBases")),
                "at_bats": _n(bat.get("atBats")),
                "plate_appearances": _n(bat.get("plateAppearances")),
                "strikeouts_batting": _n(bat.get("strikeOuts")),
                # pitching
                "innings": innings(pit.get("inningsPitched")),
                "strikeout": _n(pit.get("strikeOuts")),
                "earned_run": _n(pit.get("earnedRuns")),
                "hit_allowed": _n(pit.get("hits")),
                "walk_allowed": _n(pit.get("baseOnBalls")),
                # `hitBatsmen` is the canonical field for batters this
                # pitcher hit. `hitByPitch` also appears in the pitching
                # block and is not reliably the same thing, so it is only a
                # fallback.
                "hbp_allowed": _n(pit.get("hitBatsmen",
                                          pit.get("hitByPitch"))),
                "complete_game": _n(pit.get("completeGames")),
                "shutout": _n(pit.get("shutouts")),
                "batters_faced": _n(pit.get("battersFaced")),
                # From the pitcher's own line first. The boxscore endpoint
                # has no decisions block, and trusting one that is not there
                # cost every winning pitcher four points on the first run.
                "win": (1.0 if _n(pit.get("wins")) > 0
                        or str(pid) in winners else 0.0),
                "pitched": 1.0 if pit else 0.0,
            })
    return rows


def _n(v) -> float:
    if v is None:
        return 0.0
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- scoring

def hitter_points(df: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    for field, weight in HITTER_SCORING.items():
        if field in df:
            pts = pts + weight * pd.to_numeric(df[field],
                                               errors="coerce").fillna(0)
    return pts.round(2)


def pitcher_points(df: pd.DataFrame) -> pd.Series:
    """Pitcher score, including the two bonuses that are conditional.

    A complete-game shutout pays the complete game AND the shutout, and a
    no-hitter pays on top of both. They are stacked rather than exclusive,
    which is easy to get backwards and worth stating.
    """
    pts = pd.Series(0.0, index=df.index)
    for field, weight in PITCHER_SCORING.items():
        if field in ("complete_game", "shutout", "no_hitter"):
            continue
        if field in df:
            pts = pts + weight * pd.to_numeric(df[field],
                                               errors="coerce").fillna(0)
    cg = pd.to_numeric(df.get("complete_game", 0), errors="coerce").fillna(0)
    so = pd.to_numeric(df.get("shutout", 0), errors="coerce").fillna(0)
    hits = pd.to_numeric(df.get("hit_allowed", 0), errors="coerce").fillna(0)
    pts = pts + PITCHER_SCORING["complete_game"] * (cg > 0)
    pts = pts + PITCHER_SCORING["shutout"] * (so > 0)
    pts = pts + PITCHER_SCORING["no_hitter"] * ((cg > 0) & (hits == 0))
    return pts.round(2)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Add `points`, using whichever rule set the row belongs to."""
    out = df.copy()
    is_pitcher = pd.to_numeric(out.get("pitched", 0),
                               errors="coerce").fillna(0) > 0
    out["is_pitcher"] = is_pitcher.astype(int)
    out["points"] = np.where(is_pitcher, pitcher_points(out),
                             hitter_points(out))
    return out


# ------------------------------------------------------------- draftkings

def slates() -> pd.DataFrame:
    """Every baseball draft group DraftKings is currently selling."""
    payload = _get(DK_CONTESTS)
    contests = payload.get("Contests") or []
    if not contests:
        raise Unavailable("the baseball lobby listed no contests")
    rows = {}
    for c in contests:
        dg = c.get("dg")
        if not dg:
            continue
        r = rows.setdefault(dg, {"draft_group": dg, "contests": 0,
                                 "game_type": c.get("gameType"),
                                 "starts_text": c.get("sdstring"),
                                 "biggest_prize": 0, "biggest_field": 0,
                                 "example": c.get("n")})
        r["contests"] += 1
        r["biggest_prize"] = max(r["biggest_prize"], c.get("po") or 0)
        r["biggest_field"] = max(r["biggest_field"], c.get("m") or 0)
    out = pd.DataFrame(rows.values())
    log.info("lobby: %d draft groups, %d contests",
             len(out), int(out["contests"].sum()))
    return out.sort_values("contests", ascending=False).reset_index(drop=True)


def board(draft_group: int) -> pd.DataFrame:
    """Who is priced on a slate.

    The same lobby endpoint the football build uses, because api.draftkings
    .com answers 403 to GitHub's runners. Position is `pn`; `pp` is an
    integer that is zero for everybody, and mapping position to it once
    produced a board where every player shared one position and every
    positional constraint was vacuous.
    """
    payload = _get(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        raise Unavailable(f"draft group {draft_group} returned no players "
                          f"(top-level keys: {sorted(payload)[:10]})")
    rows = []
    for p in raw:
        home, away = p.get("htabbr"), p.get("atabbr")
        tid, htid, atid = p.get("tid"), p.get("htid"), p.get("atid")
        rows.append({
            "dk_player_id": p.get("pid"),
            # DraftKings carries the MLB id, which is the whole point: the
            # join is an integer comparison, not a name match.
            "mlb_id": _first_id(p),
            "name": " ".join(x for x in (p.get("fn"), p.get("ln")) if x),
            "position": p.get("pn"),
            "team": home if tid == htid else away if tid == atid else None,
            "opponent": away if tid == htid else home if tid == atid else None,
            "is_home": 1 if tid == htid else 0 if tid == atid else np.nan,
            "salary": pd.to_numeric(p.get("s"), errors="coerce"),
            "dk_points_per_game": pd.to_numeric(p.get("ppg"),
                                                errors="coerce"),
            "disabled": bool(p.get("IsDisabledFromDrafting")),
            "game": f"{away} @ {home}" if home and away else None,
        })
    df = pd.DataFrame(rows)
    if df["position"].nunique() <= 1:
        raise Unavailable(
            f"every player on draft group {draft_group} has position "
            f"{df['position'].iloc[0]!r}. The position field moved.")
    if df["salary"].notna().sum() == 0:
        raise Unavailable(f"draft group {draft_group}: no player has a "
                          f"salary. The salary field moved.")
    log.info("draft group %s: %d players, %d teams, $%s-$%s",
             draft_group, len(df), df["team"].nunique(),
             int(df["salary"].min()), int(df["salary"].max()))
    return df.reset_index(drop=True)


# `pdkid` first, because the field dump showed it holding a real MLB id -
# 658796 for Jacob Misiorowski, where `pid` is DraftKings' own 1217479 and
# `tsid` is a third party's. Seven guessed names missed it because none of
# them guessed that a field called "player DK id" would carry the LEAGUE's
# id. Printing the payload found in one line what guessing had not.
#
# Order matters: the first field holding anything wins, so the one known to
# be right leads and the rest stay only as fallbacks.
_ID_FIELDS = ("pdkid", "mlbId", "MlbId", "mlbid", "sportsRadarId",
              "playerId", "externalId", "srid")


def board_row_keys(draft_group: int) -> str:
    """Every key DraftKings actually sends for a player, printed.

    Seven guessed field names found a league id on zero of 652 rows. Rather
    than guess an eighth, this prints what is really there - the same move
    that turned the box-score question from three rounds into one line.
    """
    payload = _get(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        return "no players on that draft group"
    lines = [f"DraftKings sends {len(raw[0])} fields per player:"]
    for k, v in sorted(raw[0].items()):
        lines.append(f"    {k:<22} {str(v)[:44]}")
    return "\n".join(lines)


def _first_id(p: dict):
    """DraftKings' copy of the league's own player id, wherever it lives.

    Field naming in this feed has moved before, so several are tried and the
    verifier reports how many rows found one. If none does, the join falls
    back to names and the build says so loudly rather than quietly getting
    worse.
    """
    for f in _ID_FIELDS:
        v = p.get(f)
        if v not in (None, "", 0):
            return str(v)
    return None


def normalise(name) -> str:
    s = str(name or "").lower().strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    s = s.replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return " ".join(s.split())


def attach_history(board_df: pd.DataFrame, hist: pd.DataFrame
                   ) -> pd.DataFrame:
    """Join the board to history. By id where possible, by name otherwise."""
    out = board_df.copy()
    ids = set(hist["player_id"].astype(str))
    matched, how = [], []
    by_name = {}
    for pid, nm in zip(hist["player_id"].astype(str), hist["name"]):
        by_name.setdefault(normalise(nm), pid)
    for mlb_id, nm in zip(out["mlb_id"], out["name"]):
        if mlb_id is not None and str(mlb_id) in ids:
            matched.append(str(mlb_id))
            how.append("id")
            continue
        hit = by_name.get(normalise(nm))
        matched.append(hit)
        how.append("name" if hit else "none")
    out["player_id"] = pd.Series(matched, index=out.index, dtype=object)
    out["matched_by"] = how
    n = sum(1 for m in matched if m is not None)
    log.info("join: %d of %d matched (%d by id, %d by name)",
             n, len(out), how.count("id"), how.count("name"))
    return out
