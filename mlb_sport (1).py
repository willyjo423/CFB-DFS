"""Baseball, declared. Two specs, because it is two sports.

Hitters and pitchers do not share a scoring system, a roster slot, a feature
set, or even the sign of their correlation with each other. Giving them one
spec with a position flag would ask a single fitted model to learn that a
strikeout is worth +2 to the man throwing it and roughly -0.3 to the man
taking it, from the same column. So: two specs, two fitted models, and the
engine runs each without knowing there is another.

What baseball needs that football did not
-----------------------------------------
**A longer memory.** Football's four-game half-life is tuned to a
sixteen-game season where a role change is the whole story. Over 162 games a
hitter's true talent moves slowly and the noise is enormous, so the memory is
much longer - about five weeks of games. A short half-life here would chase
variance and call it form.

**Opposite correlation signs.** A pitcher is strongly NEGATIVELY correlated
with every opposing hitter: his good day IS their bad one. Football has
nothing like that, which is why the loadings live with the sport.

**Batting order, which is not here yet.** It is the single biggest driver of
a hitter's day and is published about two hours before first pitch. That is a
live-projection problem rather than a historical one, and it is the most
valuable thing still missing from this model - stated here rather than
discovered as a weak grade later.

**Two games in one day.** Football never has this. Baseball does, about
thirty times a season, and it breaks the engine's quiet assumption that a
period identifies a game. See `to_canonical`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine import SportSpec
from engine import features as EF

log = logging.getLogger(__name__)

# How many games one date can hold. Two: a doubleheader. Everything that
# converts between a day and a period goes through this, so the packing is
# stated once.
SLOTS = 2

# 162 games. A hitter's line moves slowly and its noise is enormous, so the
# past keeps mattering far longer than it does over a football season.
HALFLIFE = 30.0

HITTERS = SportSpec(
    name="MLB-hitters",
    usage=["plate_appearances", "at_bats", "single", "double", "triple",
           "home_run", "rbi", "run", "walk", "stolen_base"],
    shares=["plate_appearances"],
    touch_columns=["plate_appearances"],
    positions=["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"],
    min_prior_games=10,
    halflife=HALFLIFE,
    min_train_rows=2000,
    min_played_rows=1500,
    # Where he is batting TONIGHT, and where he usually bats.
    #
    # `bat_slot` is deliberately NOT shifted, and that is not a leak. Every
    # other feature here describes games already played, because the thing it
    # predicts has not happened yet - but the lineup card is published about
    # two hours before first pitch, so tonight's slot is genuinely known
    # before tonight's game. It is the same class of feature as a market line
    # in football: forward-looking, and the only input in the set that knows
    # something the player's own history cannot.
    #
    # `ewm_bat_slot` comes along automatically as a usage column would not -
    # it is listed here so the model can see the DIFFERENCE between where a
    # man usually bats and where he is batting today, which is exactly the
    # information a promotion or a demotion carries.
    # `opp_sp_*` is the opposing STARTER's form coming into tonight. Like
    # `bat_slot` it is forward-looking and that is legitimate: the probable
    # pitcher is announced the day before, so it is known before the game in
    # exactly the way a market line is. Unlike bat_slot it describes somebody
    # else entirely, which is the whole point - it is the first feature in
    # this spec that is not about the hitter.
    extra_features=["bat_slot", "ewm_bat_slot", "bat_started",
                    "opp_sp_k_rate", "opp_sp_baserunners"],
    loadings={
        # A batting order is a queue: team-mates score in the same innings
        # off the same pitcher, so the team term is strong and the
        # competition term is near zero. Nine hitters do not take each
        # other's plate appearances the way two running backs take carries.
        "game":    {p: 0.18 for p in
                    ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]},
        "team":    {p: 0.42 for p in
                    ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]},
        "compete": {p: 0.05 for p in
                    ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]},
    },
)

PITCHERS = SportSpec(
    name="MLB-pitchers",
    usage=["innings", "strikeout", "batters_faced", "earned_run",
           "hit_allowed", "walk_allowed"],
    shares=[],
    touch_columns=["batters_faced"],
    positions=["SP", "RP", "P"],
    min_prior_games=5,          # a starter makes ~32 appearances a season
    halflife=8.0,               # far fewer appearances, so fewer to remember
    min_train_rows=1000,
    min_played_rows=800,
    loadings={
        # A pitcher shares little with his own side's hitters and a great
        # deal with the game itself. Two pitchers on one team in one game are
        # genuinely competing - innings one throws are innings the other does
        # not - which is the one place baseball looks like football.
        "game":    {"SP": 0.30, "RP": 0.22, "P": 0.26},
        "team":    {"SP": 0.12, "RP": 0.12, "P": 0.12},
        "compete": {"SP": 0.55, "RP": 0.40, "P": 0.45},
    },
)


def to_canonical(hist: pd.DataFrame) -> pd.DataFrame:
    """Box-score rows, renamed to the engine's vocabulary.

    `period` is a day index rather than a week, because baseball has no
    weeks. Anything monotonic within a season works, and calling a date a
    week would have every sport's code quietly lying about what it orders by.

    Why a period is half a day
    --------------------------
    A period has to identify a GAME, not a date. Every other sport in this
    project gets away with conflating the two; baseball does not, because a
    doubleheader puts two games on one date. Using the raw day of year there:

      * the same player gets two rows with the same key, which is the
        duplicate-key failure the frame validator exists to catch;
      * and, far worse because nothing raises, every `share_*` feature is
        computed by grouping on (team, season, period) - so a player's share
        of his team's hits would be measured against TWO games' worth of team
        totals and come out at roughly half its true value, on exactly the
        days when a hitter has double the usual opportunity.

    So a day holds `SLOTS` periods, and a team's games within a date are
    ordered by game id. A single-game day uses slot 0 and nothing changes.
    Grading still speaks in days; `mlb_run_grade` multiplies.
    """
    out = hist.copy()
    missing = [c for c in ("player_id", "team", "points") if c not in out]
    if missing:
        raise ValueError(f"MLB history is missing {missing}")
    out["player_id"] = out["player_id"].astype(str)

    dates = pd.to_datetime(out["date"], errors="coerce")
    out["season"] = dates.dt.year.fillna(0).astype(int)
    doy = dates.dt.dayofyear.fillna(0).astype(int)

    # Rank on the game id, numerically - game_pk is carried as text so that
    # it survives the CSV cache, and ranking text would be lexicographic.
    keys = pd.DataFrame({
        "team": out["team"].astype(str),
        "season": out["season"],
        "doy": doy,
        "gpk": pd.to_numeric(out.get("game_pk"), errors="coerce"),
    })
    slot = (keys.groupby(["team", "season", "doy"])["gpk"]
                .rank(method="dense", na_option="top")
                .fillna(1).astype(int) - 1)
    # A third game on one date does not happen in modern baseball. If it ever
    # does, it collides into slot 1 and the validator says so out loud rather
    # than this quietly inventing a period that belongs to the next day.
    slot = slot.clip(lower=0, upper=SLOTS - 1)

    out["period"] = doy * SLOTS + slot
    n_second = int((slot > 0).sum())
    if n_second:
        log.info("doubleheaders: %d rows are the second game of a date",
                 n_second)
    return out


def day_of(period: int) -> int:
    """The calendar day a period falls on. The inverse of the packing above."""
    return int(period) // SLOTS


def periods_of(day: int) -> tuple[int, int]:
    """Every period belonging to one calendar day, as (first, last)."""
    return int(day) * SLOTS, int(day) * SLOTS + SLOTS - 1


def split(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two populations, separated the way the scoring already separates.

    `is_pitcher` is set by the data layer from whether a pitching line
    exists, which is the same test the scorer used. Deriving it twice from
    different rules is how the two halves drift apart.
    """
    df = to_canonical(hist)
    is_pit = pd.to_numeric(df.get("is_pitcher", 0),
                           errors="coerce").fillna(0) > 0
    pitchers = df[is_pit].copy()
    hitters = df[~is_pit].copy()
    log.info("split: %d hitter rows, %d pitcher rows",
             len(hitters), len(pitchers))
    return hitters, pitchers


def _with_order(built: pd.DataFrame) -> pd.DataFrame:
    """Add where a man USUALLY bats, alongside where he is batting today.

    Deliberately not folded into the spec's usage columns. Usage is coerced
    with `fillna(0)`, and a batting slot of zero is not a missing slot - it is
    a tenth place in the order that does not exist, and the model would learn
    from it. A pitcher, who has no slot at all, would be handed one.

    So the rolling version is built here, where a missing slot can stay
    missing. The shift comes from the engine's own `ewm`, so this cannot
    reintroduce the leak by forgetting it.
    """
    out = built.sort_values(["player_id", "season", "period"]).reset_index(
        drop=True)
    for c in ("bat_slot", "bat_started"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    out["ewm_bat_slot"] = (out.groupby("player_id", sort=False)["bat_slot"]
                           .transform(lambda s: EF.ewm(s, HALFLIFE)))
    have = float(out["bat_slot"].notna().mean()) if len(out) else 0.0
    log.info("batting order present on %.0f%% of rows", 100 * have)
    if have < 0.5:
        log.warning("fewer than half the rows carry a batting slot. If this "
                    "history was fetched before the box-score parser learned "
                    "to read `battingOrder`, re-fetch it - the feature will "
                    "otherwise be mostly missing and worth nothing.")
    return out


# --------------------------------------------------------------------------
# Who the hitter is facing
# --------------------------------------------------------------------------
# The largest thing missing from the hitter model, and it needs no new data.
#
# Every feature above describes the HITTER: how often he bats, where in the
# order, what he has done lately. None of them describe the man throwing the
# ball. So a good hitter against a Cy Young winner and the same hitter
# against a bullpen game were the same projection - in a sport where the
# opposing starter is the biggest thing separating one night from another
# after opportunity itself.
#
# It is derivable from history already fetched. Pitcher rows carry `game_pk`
# and `team`; hitter rows carry `game_pk` and `opponent`. The starter is the
# pitcher who faced the most batters in that game for that team. A real
# appearance flag would be better and is not in the box score as parsed;
# "faced the most batters" identifies the starter on every game that is not
# an opener, and an opener is a genuinely different thing to face anyway.
#
# Two rates rather than one, because they say different things to a hitter.
# A strikeout is an out with no ball in play, which cuts his floor; walks and
# hits allowed raise the ceiling of everyone batting behind him. A pitcher
# can be high in both and the pair separates him from one who is high in
# neither.
SP_HALFLIFE = 8.0


def _starter_rows(hist: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, team): the man who started that game for them."""
    _, pitchers = split(hist)
    if pitchers.empty or "game_pk" not in pitchers.columns:
        return pd.DataFrame()
    p = pitchers.copy()
    for c in ("batters_faced", "strikeout", "hit_allowed", "walk_allowed"):
        p[c] = pd.to_numeric(p.get(c), errors="coerce").fillna(0.0)
    p["game_pk"] = p["game_pk"].astype(str)
    p = p.sort_values(["game_pk", "team", "batters_faced"],
                      ascending=[True, True, False])
    s = p.drop_duplicates(["game_pk", "team"], keep="first").copy()
    faced = s["batters_faced"].replace(0, np.nan)
    s["_k"] = s["strikeout"] / faced
    s["_br"] = (s["hit_allowed"] + s["walk_allowed"]) / faced
    return s.sort_values(["player_id", "season", "period"]).reset_index(
        drop=True)


def starter_quality(hist: pd.DataFrame) -> pd.DataFrame:
    """Each start, with that starter's form BEFORE it.

    Keyed by (game_pk, team) - the team he was throwing FOR - so it joins
    onto a hitter row by that hitter's OPPONENT.

    The rates go through `EF.ewm`, which shifts by one, so a start cannot see
    itself. That function is used rather than reimplemented here precisely so
    this cannot become the place the leak comes back.
    """
    s = _starter_rows(hist)
    if s.empty:
        return pd.DataFrame(columns=["game_pk", "team", "opp_sp_id",
                                     "opp_sp_k_rate", "opp_sp_baserunners"])
    g = s.groupby("player_id", sort=False)
    s["opp_sp_k_rate"] = g["_k"].transform(lambda x: EF.ewm(x, SP_HALFLIFE))
    s["opp_sp_baserunners"] = g["_br"].transform(
        lambda x: EF.ewm(x, SP_HALFLIFE))
    out = s[["game_pk", "team", "player_id", "opp_sp_k_rate",
             "opp_sp_baserunners"]].rename(columns={"player_id": "opp_sp_id"})
    log.info("starters: %d starts, %d carrying prior form",
             len(out), int(out["opp_sp_k_rate"].notna().sum()))
    return out


def current_starter_form(hist: pd.DataFrame) -> pd.DataFrame:
    """Every pitcher's form as of NOW, for a slate not yet played.

    `starter_quality` answers "what had he done before that game". This
    answers "what has he done before tonight", which is the same quantity one
    appearance further along - so here the EWM is deliberately NOT shifted.
    Shifting would throw away his most recent start, which is the one piece
    of evidence a board most wants.
    """
    s = _starter_rows(hist)
    if s.empty:
        return pd.DataFrame(columns=["opp_sp_id", "opp_sp_k_rate",
                                     "opp_sp_baserunners"])
    g = s.groupby("player_id", sort=False)
    s["opp_sp_k_rate"] = g["_k"].transform(
        lambda x: x.ewm(halflife=SP_HALFLIFE, min_periods=1).mean())
    s["opp_sp_baserunners"] = g["_br"].transform(
        lambda x: x.ewm(halflife=SP_HALFLIFE, min_periods=1).mean())
    last = s.drop_duplicates("player_id", keep="last")
    return last[["player_id", "opp_sp_k_rate",
                 "opp_sp_baserunners"]].rename(
        columns={"player_id": "opp_sp_id"})


def _with_opponent_starter(built: pd.DataFrame,
                           hist: pd.DataFrame) -> pd.DataFrame:
    """Attach the opposing starter's prior form to every hitter row."""
    out = built.copy()
    q = starter_quality(hist)
    if q.empty or "game_pk" not in out.columns:
        for c in ("opp_sp_k_rate", "opp_sp_baserunners"):
            out[c] = np.nan
        log.warning("no starter form could be derived - the hitter model is "
                    "blind to who is pitching")
        return out

    n_in = len(out)
    out["game_pk"] = out["game_pk"].astype(str)
    out = out.merge(q.drop(columns=["opp_sp_id"]).rename(
        columns={"team": "opponent"}),
        on=["game_pk", "opponent"], how="left", validate="m:1")
    if len(out) != n_in:
        raise SystemExit(
            f"joining the opposing starter changed the row count "
            f"({n_in} -> {len(out)}), so (game_pk, team) is not unique on "
            f"the starter table. Fix that before trusting any projection.")
    have = float(out["opp_sp_k_rate"].notna().mean()) if len(out) else 0.0
    log.info("opposing starter known on %.0f%% of hitter rows", 100 * have)
    if have < 0.5:
        log.warning("fewer than half the hitter rows know who they faced, so "
                    "this feature is mostly missing and worth little.")
    return out


def build(hist: pd.DataFrame, which: str) -> pd.DataFrame:
    """Features for one half of the sport."""
    hitters, pitchers = split(hist)
    if which == "hitters":
        return _with_opponent_starter(
            _with_order(EF.build(hitters, HITTERS)), hist)
    if which == "pitchers":
        return EF.build(pitchers, PITCHERS)
    raise ValueError(f"which must be 'hitters' or 'pitchers', got {which!r}")


SPECS = {"hitters": HITTERS, "pitchers": PITCHERS}
