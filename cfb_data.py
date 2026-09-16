"""College football data: the board, ten seasons of history, and the join.

This is the CFB counterpart to the NFL project's data.py, and it is a rewrite
rather than a port because every input differs. What carries over is the set
of mistakes already paid for, each of which is prevented here by construction:

* **The board comes from the lobby, not the API.** api.draftkings.com answers
  403 to GitHub runners. www.draftkings.com/lineup/getavailableplayers answers
  200 with the whole slate, in an abbreviated dialect - fn/ln, s, pn.

* **Position is `pn`, never `pp`.** `pp` is an integer that is zero for every
  player on the board. Mapping position to it once produced 850 players
  sharing a single position called `0`, which makes every positional
  constraint vacuous and a lineup of eight quarterbacks perfectly legal.

* **Team codes do not join; fixtures do.** DraftKings writes BAMA, TA&M, RU;
  CFBD writes Alabama, Texas A&M, Rutgers. Two of twenty-four matched by
  string. `UL` is Louisiana to DraftKings and reads equally well as
  Louisville, and nothing about the string settles it - but a fixture must
  match exactly one CFBD fixture on BOTH sides, and that does settle it.

* **Names need two spellings, not one.** DraftKings writes "AJ Swann", CFBD
  writes "A.J. Swann". Period-to-space gives "aj swann" and "a j swann",
  which do not match; period-to-nothing breaks "St.Brown" instead. Both rules
  are right and they contradict, so both keys are carried and either may hit.

* **The live week is fetched, not calculated.** Probing showed the season's
  completed-game count equal to weeks 1 and 2 exactly while week 3 sat
  scheduled with zero played. A clock-based guess would have asked for a week
  that does not exist yet and quietly returned nothing.

* **Captain salary is charged, not assumed.** On Showdown the captain costs
  1.5x salary as well as scoring 1.5x points. Reading only the points
  multiplier builds lineups that cannot be entered.
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

CFBD = "https://api.collegefootballdata.com"
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=CFB"
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")

# DraftKings college football. Same shape as the NFL game with one difference
# that matters for pricing: college is full PPR, so slot receivers on
# pass-happy teams are worth more than their yardage suggests.
SCORING = {
    "pass_yards": 0.04, "pass_td": 4.0, "interception": -1.0,
    "rush_yards": 0.1, "rush_td": 6.0,
    "rec": 1.0, "rec_yards": 0.1, "rec_td": 6.0,
    "return_td": 6.0, "fumble_lost": -1.0,
}
# The bonuses are why a projection has to be a distribution and not a mean. A
# 95-yard rusher and a 105-yard rusher are one yard of talent apart and three
# points apart, and only a simulation prices that step.
BONUS = {"pass_yards": (300, 3.0), "rush_yards": (100, 3.0),
         "rec_yards": (100, 3.0)}

STAT_MAP = {
    ("passing", "YDS"): "pass_yards", ("passing", "TD"): "pass_td",
    ("passing", "INT"): "interception", ("passing", "C/ATT"): "completions",
    ("rushing", "YDS"): "rush_yards", ("rushing", "TD"): "rush_td",
    ("rushing", "CAR"): "carries",
    ("receiving", "YDS"): "rec_yards", ("receiving", "TD"): "rec_td",
    ("receiving", "REC"): "rec",
    ("fumbles", "LOST"): "fumble_lost",
    ("kickReturns", "TD"): "kick_return_td",
    ("puntReturns", "TD"): "punt_return_td",
}

STAT_FIELDS = sorted(set(STAT_MAP.values()))

# Seasons the depth probe cleared. 2014-2015 lack fumbles/LOST entirely, and
# scoring them would credit every fumble as zero rather than -1 - wrong in a
# way that looks fine. 2020 is present but is a different sport: 20 games in
# week 3 against a normal 65, opt-outs, and a schedule rewritten weekly.
USABLE_FIRST = 2016
COVID_SEASON = 2020

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


class Unavailable(RuntimeError):
    """A source this build cannot proceed without did not answer."""


# --------------------------------------------------------------------- names

def name_keys(name) -> set[str]:
    """Every spelling two sources might agree on, not just one.

    Deliberately not fuzzy. Every key is an exact reduction, because an
    edit-distance match would pair two different players with similar names,
    and in college football - where rosters run to 120 and brothers are
    common - that is a far worse failure than a miss.
    """
    raw = str(name or "").strip()
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first.strip()} {last.strip()}"
    keys = set()
    for dot in (" ", ""):
        s = raw.lower().replace("-", " ").replace(".", dot)
        s = s.replace("'", "").replace("`", "")
        s = _SUFFIXES.sub(" ", s)
        s = " ".join(s.split())
        if s:
            keys.add(s)
    return keys


def primary_key(name) -> str:
    """One key, for grouping. The period-to-space spelling."""
    keys = name_keys(name)
    return sorted(keys)[0] if keys else ""


# ----------------------------------------------------------------- transport

def _get_json(url: str, headers=None, params=None):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=headers or UA, params=params,
                             timeout=TIMEOUT)
        except Exception as exc:                   # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:100]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as exc:               # noqa: BLE001
                raise Unavailable(
                    f"{url[:70]} returned 200 but not JSON "
                    f"({type(exc).__name__}): {r.content[:100]!r}") from exc
        # 429 and 5xx are worth another try; 4xx is not.
        last = f"HTTP {r.status_code}: {r.text[:100]}"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        break
    raise Unavailable(f"{url[:70]} -> {last}")


def cfbd(path: str, key: str, **params):
    hdr = dict(UA)
    hdr["Authorization"] = f"Bearer {key}"
    return _get_json(f"{CFBD}/{path}", headers=hdr, params=params)


# ---------------------------------------------------------------- draftkings

def slates() -> pd.DataFrame:
    """Every college football draft group DraftKings is currently selling."""
    payload = _get_json(DK_CONTESTS)
    contests = payload.get("Contests") or []
    if not contests:
        raise Unavailable("the college football lobby listed no contests")
    rows = {}
    for c in contests:
        dg = c.get("dg")
        if not dg:
            continue
        r = rows.setdefault(dg, {
            "draft_group": dg, "contests": 0,
            "game_type": c.get("gameType"),
            "starts_text": c.get("sdstring"),
            "biggest_prize": 0, "biggest_field": 0,
            "example": c.get("n"),
        })
        r["contests"] += 1
        r["biggest_prize"] = max(r["biggest_prize"], c.get("po") or 0)
        r["biggest_field"] = max(r["biggest_field"], c.get("m") or 0)
    out = pd.DataFrame(rows.values())
    log.info("lobby: %d draft groups, %d contests",
             len(out), int(out["contests"].sum()))
    return out.sort_values("contests", ascending=False).reset_index(drop=True)


def board(draft_group: int, captain_multiplier: float | None = None
          ) -> pd.DataFrame:
    """Who is priced on a slate.

    `charged_salary` is what the cap actually sees. On Showdown the captain
    costs 1.5x as well as scoring 1.5x, and a solver that charges face value
    builds lineups DraftKings will not accept.
    """
    payload = _get_json(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        raise Unavailable(f"draft group {draft_group} returned no players "
                          f"(top-level keys: {sorted(payload)[:10]})")

    rows = []
    for p in raw:
        home, away = p.get("htabbr"), p.get("atabbr")
        tid, htid, atid = p.get("tid"), p.get("htid"), p.get("atid")
        team = home if tid == htid else away if tid == atid else None
        opponent = away if tid == htid else home if tid == atid else None
        rows.append({
            "dk_player_id": p.get("pid"),
            "name": " ".join(x for x in (p.get("fn"), p.get("ln")) if x),
            "position": p.get("pn"),
            "team": team,
            "opponent": opponent,
            "is_home": 1 if tid == htid else 0 if tid == atid else np.nan,
            "salary": pd.to_numeric(p.get("s"), errors="coerce"),
            "dk_points_per_game": pd.to_numeric(p.get("ppg"), errors="coerce"),
            "disabled": bool(p.get("IsDisabledFromDrafting")),
            "roster_slot": p.get("rosposid"),
            "game": f"{away} @ {home}" if home and away else None,
        })
    df = pd.DataFrame(rows)

    # A board where every player shares one position, or where every salary is
    # NaN, is a mapping failure that looks exactly like a successful parse.
    if df["position"].nunique() <= 1:
        raise Unavailable(
            f"every player on draft group {draft_group} has position "
            f"{df['position'].iloc[0]!r}. The position field moved; look at a "
            f"raw row before trusting anything downstream.")
    if df["salary"].notna().sum() == 0:
        raise Unavailable(f"draft group {draft_group}: no player has a "
                          f"salary. The salary field moved.")

    if df["team"].isna().any():
        n = int(df["team"].isna().sum())
        log.warning("%d of %d players could not be assigned to a side of "
                    "their fixture - dropped rather than guessed", n, len(df))
        df = df.dropna(subset=["team"])

    # Showdown: DraftKings lists the captain as a separate, pre-multiplied
    # row. Detect it from the prices themselves rather than from the slate's
    # name, because the name is a label and the prices are the contract.
    df["is_captain"] = 0
    if captain_multiplier:
        df = _mark_captains(df, captain_multiplier)
    df["charged_salary"] = df["salary"] * np.where(
        df["is_captain"] == 1, captain_multiplier or 1.0, 1.0)

    df["keys"] = df["name"].map(name_keys)
    df["key"] = df["name"].map(primary_key)

    log.info("draft group %s: %d players, %d teams, %d games, $%s-$%s",
             draft_group, len(df), df["team"].nunique(), df["game"].nunique(),
             int(df["salary"].min()), int(df["salary"].max()))
    return df.reset_index(drop=True)


def _mark_captains(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    """Flag rows DraftKings has already multiplied.

    On a Showdown board the same player appears twice - once at face value and
    once at roughly `mult` times it. Pairing by name and comparing the two
    prices identifies the captain row without trusting a slot id whose meaning
    changes between sports.
    """
    out = df.copy()
    by_key = out.groupby(out["name"].map(primary_key))["salary"]
    lo = by_key.transform("min")
    ratio = out["salary"] / lo.replace(0, np.nan)
    out["is_captain"] = ((ratio > (mult + 1.0) / 2.0) & (ratio.notna())
                         ).astype(int)
    n = int(out["is_captain"].sum())
    if n:
        log.info("marked %d captain rows (salary ~%.2fx the base row)",
                 n, mult)
    return out


# --------------------------------------------------------------------- cfbd

def live_week(key: str, season: int) -> int:
    """The newest week that has actually been played, fetched not guessed.

    Asking the schedule which games are COMPLETE is the only reliable answer.
    A clock-based estimate gets this wrong in both directions - byes, Tuesday
    MACtion, and a season whose week 1 does not sit where the calendar says.
    """
    played = 0
    for week in range(1, 20):
        try:
            games = cfbd("games", key, year=season, week=week,
                         seasonType="regular")
        except Unavailable as exc:
            log.warning("week %d schedule unavailable: %s", week, exc)
            break
        if not games:
            break
        done = sum(1 for g in games
                   if g.get("completed") is True
                   or (g.get("home_points") is not None
                       and g.get("away_points") is not None))
        if done == 0:
            break
        played = week
    if played == 0:
        raise Unavailable(f"no completed {season} games in any week")
    log.info("%d: newest completed week is %d", season, played)
    return played


def roster_positions(key: str, season: int) -> pd.DataFrame:
    """athlete_id -> position for a season.

    games/players carries no position and the roster endpoint is the only
    source of one. Blank positions are dropped rather than kept as empty
    strings: a row that matched an entry with no position is not a match, and
    counting it as one overstates coverage by about six points.
    """
    payload = cfbd("roster", key, year=season)
    if not payload:
        raise Unavailable(f"{season} roster is empty")
    sample = payload[0]
    id_field = next((f for f in ("id", "athlete_id", "athleteId", "player_id")
                     if f in sample), None)
    if id_field is None:
        raise Unavailable(f"{season} roster has no id field; "
                          f"keys are {sorted(sample)[:12]}")
    rows = [{"athlete_id": str(p[id_field]),
             "position": str(p.get("position") or "").upper().strip(),
             "roster_name": " ".join(
                 x for x in (p.get("first_name") or p.get("firstName"),
                             p.get("last_name") or p.get("lastName")) if x)}
            for p in payload if p.get(id_field) is not None]
    df = pd.DataFrame(rows)
    blank = int((df["position"] == "").sum())
    df = df[df["position"] != ""].drop_duplicates("athlete_id")
    log.info("%d roster: %d positioned athletes (%d blank, dropped)",
             season, len(df), blank)
    return df.reset_index(drop=True)


def _number(value):
    """A stat as a number. CFBD sends every value as a string.

    `C/ATT` arrives as "18/25", which float() rejects; the part before the
    slash is completions. A bare float() would turn every quarterback's
    completion count into NaN, and NaN is not zero - it propagates into the
    feature and deletes the row.
    """
    if value is None:
        return np.nan
    s = str(value).strip().replace(",", "")
    if "/" in s:
        s = s.split("/")[0]
    try:
        return float(s)
    except ValueError:
        return np.nan


def flatten_player_games(payload: list[dict]) -> pd.DataFrame:
    """One row per player per game, columns named for the scoring rule."""
    rows = []
    for game in payload or []:
        for team in game.get("teams") or []:
            school = team.get("team") or team.get("school")
            opponent = None
            for other in game.get("teams") or []:
                if (other.get("team") or other.get("school")) != school:
                    opponent = other.get("team") or other.get("school")
            homeaway = team.get("homeAway")
            for cat in team.get("categories") or []:
                cname = cat.get("name")
                for typ in cat.get("types") or []:
                    field = STAT_MAP.get((cname, typ.get("name")))
                    if field is None:
                        continue
                    for ath in typ.get("athletes") or []:
                        rows.append({
                            "game_id": game.get("id"), "school": school,
                            "opponent": opponent,
                            "is_home": 1 if homeaway == "home" else 0,
                            "athlete_id": str(ath.get("id")),
                            "name": ath.get("name"),
                            "field": field,
                            "value": _number(ath.get("stat")),
                        })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    wide = (long.pivot_table(
        index=["game_id", "school", "opponent", "is_home",
               "athlete_id", "name"],
        columns="field", values="value", aggfunc="sum").reset_index())
    wide.columns.name = None
    for field in STAT_FIELDS:
        if field not in wide:
            wide[field] = 0.0
    return wide.fillna({f: 0.0 for f in STAT_FIELDS})


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    """DraftKings college football points for each row."""
    pts = pd.Series(0.0, index=df.index)
    for field, weight in SCORING.items():
        if field == "return_td":
            pts = pts + weight * (df.get("kick_return_td", 0)
                                  + df.get("punt_return_td", 0))
        elif field in df:
            pts = pts + weight * df[field]
    for field, (threshold, bonus) in BONUS.items():
        if field in df:
            pts = pts + bonus * (df[field] >= threshold)
    return pts.round(2)


def season_history(key: str, season: int, through_week: int,
                   positions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every settled game in a season, scored, with positions attached.

    `through_week` is inclusive of the last PLAYED week. Callers predicting
    week W must pass W-1; including W puts a player's own result into the
    features used to predict it, which is the leak that makes a model look
    brilliant in backtest and lose money on Saturday.
    """
    if positions is None:
        positions = roster_positions(key, season)
    frames = []
    for week in range(1, through_week + 1):
        try:
            payload = cfbd("games/players", key, year=season, week=week,
                           seasonType="regular")
        except Unavailable as exc:
            log.warning("%d week %d unavailable: %s", season, week, exc)
            continue
        wide = flatten_player_games(payload)
        if wide.empty:
            log.warning("%d week %d returned no player stats", season, week)
            continue
        wide["season"], wide["week"] = season, week
        frames.append(wide)
        time.sleep(0.35)
    if not frames:
        raise Unavailable(f"no {season} player stats through week "
                          f"{through_week}")
    out = pd.concat(frames, ignore_index=True)
    out["points"] = fantasy_points(out)

    before = len(out)
    out = out.merge(positions[["athlete_id", "position"]],
                    on="athlete_id", how="left")
    unplaced = int(out["position"].isna().sum())
    out = out.dropna(subset=["position"])
    log.info("%d: %d player-games, %d athletes, dropped %d without a "
             "position (%.1f%%), points %.1f to %.1f",
             season, len(out), out["athlete_id"].nunique(), unplaced,
             100 * unplaced / max(1, before),
             out["points"].min(), out["points"].max())

    out["key"] = out["name"].map(primary_key)
    out["keys"] = out["name"].map(name_keys)
    return out


def training_history(key: str, first: int, last: int,
                     through_week: dict[int, int] | None = None,
                     skip_covid: bool = True) -> pd.DataFrame:
    """Several seasons, concatenated, for fitting.

    2020 is excluded by default. It is not a thin season - it is a different
    sport, played by teams missing opt-outs against a schedule rewritten every
    week. Including it adds rows and subtracts signal. The flag exists so the
    walk-forward can measure that claim rather than leaving it an assertion.
    """
    through_week = through_week or {}
    frames = []
    for season in range(max(first, USABLE_FIRST), last + 1):
        if skip_covid and season == COVID_SEASON:
            log.info("skipping %d (covid season)", COVID_SEASON)
            continue
        try:
            positions = roster_positions(key, season)
            weeks = through_week.get(season)
            if weeks is None:
                weeks = live_week(key, season) if season == last else 15
            frames.append(season_history(key, season, weeks, positions))
        except Unavailable as exc:
            log.warning("season %d skipped: %s", season, exc)
    if not frames:
        raise Unavailable(f"no usable seasons between {first} and {last}")
    out = pd.concat(frames, ignore_index=True)
    log.info("training history: %d player-games across %d seasons "
             "(%s), %d athletes",
             len(out), out["season"].nunique(),
             ", ".join(str(s) for s in sorted(out["season"].unique())),
             out["athlete_id"].nunique())
    return out


# ----------------------------------------------------------------- the join

def attach_history(board_df: pd.DataFrame, hist: pd.DataFrame
                   ) -> pd.DataFrame:
    """Give every priced player his own scored games, or nothing.

    Matching is on EITHER spelling. The index is built from the history side's
    key set so a DraftKings spelling that differs only in periods still lands.
    """
    index: dict[str, str] = {}
    for athlete, keys in zip(hist["athlete_id"], hist["keys"]):
        for k in keys:
            index.setdefault(k, athlete)

    matched = []
    for keys in board_df["keys"]:
        hit = next((index[k] for k in keys if k in index), None)
        matched.append(hit)
    out = board_df.copy()
    out["athlete_id"] = matched

    n = int(out["athlete_id"].notna().sum())
    log.info("join: %d of %d priced players matched to history (%.1f%%)",
             n, len(out), 100 * n / max(1, len(out)))
    return out


def join_quality(joined: pd.DataFrame) -> str:
    """What the misses cost, in salary terms rather than in count.

    A join is allowed to fail at the bottom of the price range, where a
    missing projection costs nothing because nobody was rostering them. It is
    not allowed to fail on an $8,500 quarterback. Counting misses cannot tell
    those apart; pricing them can.
    """
    miss = joined[joined["athlete_id"].isna()]
    hit = joined[joined["athlete_id"].notna()]
    if miss.empty:
        return "join: every priced player matched"
    lines = [f"join: {len(miss)} of {len(joined)} unmatched "
             f"({100*len(miss)/len(joined):.1f}%)",
             f"  unmatched salary: ${miss['salary'].min():,.0f}-"
             f"${miss['salary'].max():,.0f}, median "
             f"${miss['salary'].median():,.0f}",
             f"  matched   salary: ${hit['salary'].min():,.0f}-"
             f"${hit['salary'].max():,.0f}, median "
             f"${hit['salary'].median():,.0f}"]
    expensive = miss[miss["salary"] >= hit["salary"].median()]
    if not expensive.empty:
        lines.append(f"  {len(expensive)} unmatched players priced at or "
                     f"above the matched median - these matter:")
        for r in expensive.nlargest(10, "salary").itertuples(index=False):
            lines.append(f"    ${int(r.salary):>6,}  {r.position:<4} "
                         f"{str(r.name)[:28]:<28} {r.team}")
    else:
        lines.append("  every miss is below the matched median price - the "
                     "shape a join is allowed to fail in")
    return "\n".join(lines)


# ------------------------------------------------------------------ fixtures

def _tokens(school: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(school).lower()) if t]


def fixture_team_map(board_df: pd.DataFrame, games: list[dict]) -> dict:
    """DraftKings team code -> CFBD school, solved from the schedule.

    A hand-written map is twenty-four guesses that look like knowledge, and at
    least one is wrong in a way nobody can see.

    `UL` is the case that shaped this function. It is Louisiana to
    DraftKings, it reads equally well as Louisville, and - the part that
    matters - NO string rule can propose it, because the letters stand for
    "University of Louisiana" and neither appears in the school name. An
    earlier version required the string to nominate candidates and therefore
    nominated none, leaving UL unmapped: not wrong, but not solved either.

    So the string does not have to propose anything. A code that matches no
    school at all is treated as UNCONSTRAINED rather than as hopeless, and the
    fixture resolves it from the other side: TXST is plainly Texas State,
    exactly one CFBD game has Texas State at home, and whoever is visiting is
    UL. Solving propagates - each code resolved tightens every fixture it
    appears in - so this runs to a fixpoint rather than in one pass.

    Anything still ambiguous at the end is reported, never guessed. A wrong
    team code does not drop a player; it hands him the wrong opponent, the
    wrong implied total and the wrong correlation group, and produces a full
    board of confident numbers that are wrong in every row.
    """
    cfbd_fixtures = []
    for g in games:
        home = g.get("home_team") or g.get("homeTeam")
        away = g.get("away_team") or g.get("awayTeam")
        if home and away:
            cfbd_fixtures.append((str(away), str(home)))
    schools = {s for fx in cfbd_fixtures for s in fx}

    dk_fixtures = []
    for game in (board_df.dropna(subset=["game"])["game"].drop_duplicates()):
        away_code, _, home_code = str(game).partition(" @ ")
        if away_code.strip() and home_code.strip():
            dk_fixtures.append((away_code.strip(), home_code.strip()))

    # A code is "free" when no school in this week's schedule resembles it.
    # UL is free; LOU is not, because it resembles both Louisiana and
    # Louisville and must stay confined to those two.
    free = {code for fx in dk_fixtures for code in fx
            if not any(_overlap(code, s) for s in schools)}
    if free:
        log.info("codes no string rule can propose, left to the fixtures: %s",
                 ", ".join(sorted(free)))

    mapping: dict[str, str] = {}

    def compatible(code: str, school: str) -> bool:
        if code in mapping:
            return mapping[code] == school
        if school in mapping.values():
            return False                  # another code already claimed it
        return code in free or _overlap(code, school)

    unsolved = list(dk_fixtures)
    for _ in range(len(dk_fixtures) + 1):
        progressed = False
        still = []
        for away_code, home_code in unsolved:
            cands = [(a, h) for a, h in cfbd_fixtures
                     if compatible(away_code, a) and compatible(home_code, h)]
            if len(cands) == 1:
                a, h = cands[0]
                if mapping.get(away_code, a) == a and \
                   mapping.get(home_code, h) == h:
                    mapping[away_code], mapping[home_code] = a, h
                    progressed = True
                    continue
            still.append((away_code, home_code))
        unsolved = still
        if not progressed:
            break

    if unsolved:
        log.warning("%d fixtures did not resolve to exactly one CFBD game: %s",
                    len(unsolved),
                    ", ".join(f"{a} @ {h}" for a, h in unsolved[:6]))
    log.info("team map: %d codes solved from %d fixtures",
             len(mapping), len(dk_fixtures))
    return mapping


def _overlap(code: str, school: str) -> bool:
    """Loose enough to propose, never loose enough to decide alone.

    This only nominates candidates; the both-sides fixture constraint is what
    actually decides. A permissive test here is safe and a strict one is not,
    because a strict test rejects BAMA/Alabama and leaves the fixture unsolved.
    """
    c = re.sub(r"[^a-z0-9]", "", str(code).lower())
    toks = _tokens(school)
    flat = "".join(toks)
    if not c or not flat:
        return False
    if c in flat or flat.startswith(c):
        return True
    initials = "".join(t[0] for t in toks)
    if c == initials:
        return True
    return any(t.startswith(c) or c.startswith(t[:4]) for t in toks if t)
