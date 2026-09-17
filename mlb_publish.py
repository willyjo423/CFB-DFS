"""One live baseball slate, turned into the file the page reads.

    python mlb_publish.py                    # the busiest live classic slate
    python mlb_publish.py --draft-group 1234
    python mlb_publish.py --field 50000

The page is the same page the football build uses, and it does the heavy work
in the browser: it rebuilds the factor model from the loadings, simulates the
slate, and solves for lineups under whatever locks and fades you set. So this
file's whole job is to emit a JSON document of exactly the shape that page
expects, plus a manifest listing the slates available.

The contract is read off the page's own JavaScript rather than guessed:

    docs/data/manifest.json
        {updated_at, slates: [{sport, site, kind, label, file}]}

    docs/data/<file>.json
        {generated_at, field_size, quantiles, loadings, roster,
         players: [{name, pos, team, game, salary, q, med, ceil, own, lev,
                    status}],
         server_lineups: {cash: {...}, gpp: {...}}}

Two positions vocabularies, kept apart on purpose
-------------------------------------------------
The history speaks box-score positions - LF, CF, RF, DH - and DraftKings
speaks roster positions, where all three outfielders are OF. Translating one
into the other before the fit would corrupt the model's own position dummies;
translating after it would leave the roster rules unenforceable.

So they never meet. The projection is made on HISTORY rows carrying history
positions, and only then joined onto the board, which keeps its own. The board
position is what reaches the page, because that is what the roster rules are
written in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

import mlb_data as MD
import mlb_sport as MS
from engine import cache as C
from engine import model as M
from engine import optimise as O
from engine import ownership as OWN
from engine import simulate as S

log = logging.getLogger("mlb_publish")

SPORT = "mlb"
DOCS = Path("docs")
DATA = DOCS / "data"

# DraftKings MLB Classic. Ten slots, $50,000, and no more than five hitters
# from any one team.
#
# Two things are deliberately STRICTER here than DraftKings actually requires,
# because a lineup that is too constrained can still be entered and one that is
# not constrained enough cannot:
#
#   * `max_per_team` is applied to every player rather than to hitters only,
#     so a lineup can never exceed the real limit;
#   * a player eligible at several positions is pinned to the first one
#     DraftKings lists, so the optimiser never uses eligibility it might have
#     read wrong.
#
# Both cost a little optimality and neither can produce a rejected entry.
ROSTER = {
    "slots": ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"],
    "flex_positions": [],
    "salary_cap": 50_000,
    "max_per_team": 5,
}

# Everyone who pitches fills a P slot, whatever DraftKings calls him.
PITCHER_CODES = {"P", "SP", "RP"}
HITTER_SLOTS = {"C", "1B", "2B", "3B", "SS", "OF"}


def roster_position(raw: str) -> str | None:
    """A DraftKings position string, reduced to the slot it can fill.

    DraftKings writes multi-eligibility with a slash - "1B/OF" - and the first
    listed position is the one used. A player whose position matches no slot
    at all is dropped rather than guessed at.
    """
    first = str(raw or "").split("/")[0].strip().upper()
    if first in PITCHER_CODES:
        return "P"
    return first if first in HITTER_SLOTS else None


def _id_text(s: pd.Series) -> pd.Series:
    """A player id as text, via a number, and never via `astype(str)`.

    A column of integers that contains a single missing value becomes float64
    in pandas, and `str(658796.0)` is "658796.0" - which does not equal
    "658796" and never will. That is the whole explanation for a join that
    matched 0 of 278 rows on an id both sides genuinely carry.

    It failed silently because the name fallback picked up 255 of them, so the
    only visible symptom was a log line nobody had to act on. This is the
    "a 96% join looks fine and is not" failure, in its exact original form.

    Missing stays MISSING - never the empty string. An empty string is a
    value, and a merge happily matches it against every other empty string on
    the far side, so one board player without an id fans out into a row for
    every projection without one. Null never matches null, which is exactly
    the behaviour wanted here.
    """
    num = pd.to_numeric(s, errors="coerce")
    return num.astype("Int64").astype("string")


def project_half(hist: pd.DataFrame, which: str) -> pd.DataFrame:
    """Fit one half of the sport and project every player's next outing."""
    spec = MS.SPECS[which]
    built = MS.build(hist, which)
    proj = M.Projections(spec).fit(built)
    latest = M.latest_rows(built)
    q = proj.predict(latest)
    out = latest[["player_id", "name", "team", "position"]].copy()
    for c in q.columns:
        out[c] = q[c].to_numpy()
    out["half"] = which
    log.info("%s: fitted on %d rows, projected %d players",
             spec.name, proj.trained_rows, len(out))
    return out


def join_board(board: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Board joined to projections on the LEAGUE's player id.

    This is an integer comparison, not a name match, and that is the whole
    reason the box-score fetch bothered to carry `mlb_id`. Names are kept only
    as a fallback and reported separately, because a name join that quietly
    works at 80% is how the players who changed teams disappear.
    """
    left = board.copy()
    left["mlb_id"] = _id_text(left["mlb_id"])
    right = proj.copy()
    right["player_id"] = _id_text(right["player_id"])

    n_in = len(left)
    merged = left.merge(right.drop(columns=["team", "name"]),
                        left_on="mlb_id", right_on="player_id", how="left")

    # A left join must not change the row count. If it does, both sides shared
    # a key that repeats - and the board would carry a player several times,
    # each copy with a different projection, which is how an optimiser ends up
    # fielding the same man twice or filling a slate with rows nobody put on
    # it. The college build lost 700 rows to 28,350 this way.
    if len(merged) != n_in:
        raise SystemExit(
            f"the projection join fanned {n_in} board rows into "
            f"{len(merged)}. Duplicate ids on one side; the board is not "
            f"trustworthy and nothing has been published.")
    by_id = int(merged["player_id"].notna().sum())

    miss = merged["player_id"].isna()
    if miss.any():
        keys = {MD.normalise(n): p for n, p in
                zip(proj["name"], proj["player_id"])}
        found = merged.loc[miss, "name"].map(
            lambda n: keys.get(MD.normalise(n)))
        take = found.notna()
        if take.any():
            fill = proj.set_index(proj["player_id"].astype(str))
            for idx, pid in found[take].items():
                row = fill.loc[str(pid)]
                for c in row.index:
                    if c in merged.columns and c not in ("team", "name"):
                        merged.at[idx, c] = row[c]
        log.info("join: %d of %d on MLB id, %d more on name",
                 by_id, len(merged), int(take.sum()))
    else:
        log.info("join: %d of %d on MLB id", by_id, len(merged))
    return merged


def slate_players(merged: pd.DataFrame, spec_quantiles: list[float],
                  own: pd.Series, lev: pd.Series) -> list[dict]:
    qcols = [f"q{int(round(q * 100)):02d}" for q in spec_quantiles]
    rows = []
    for i, r in merged.reset_index(drop=True).iterrows():
        rows.append({
            "name": str(r["name"]),
            # `position` by this point, not `slot` - the engine's own column
            # name, because the pool has already been renamed for it.
            "pos": str(r["position"]),
            "team": str(r["team"]),
            "game": str(r.get("game") or ""),
            "salary": int(r["salary"]),
            "q": [round(float(r[c]), 3) for c in qcols],
            "med": round(float(r["median"]), 2),
            "ceil": round(float(r["ceiling"]), 2),
            "own": round(float(own.iloc[i]), 5),
            "lev": round(float(lev.iloc[i]), 3),
            "status": "clear",
        })
    return rows


def loadings_for_page() -> dict:
    """One loadings table keyed by the positions the PAGE will see.

    The specs are keyed by box-score positions; the board is keyed by roster
    positions. The page looks these up by whatever `pos` each player carries in
    the JSON, so they are rewritten here into that vocabulary rather than left
    to fall through to a default that would silently flatten every correlation
    in the slate.
    """
    out: dict[str, dict[str, float]] = {}
    for kind in ("game", "team", "compete"):
        table = {}
        hit = MS.HITTERS.loadings.get(kind, {})
        # Hitter loadings do not vary by position - a catcher and a centre
        # fielder sit in the same batting order - so any one of them stands
        # for the lot.
        if hit:
            value = float(next(iter(hit.values())))
            for slot in sorted(HITTER_SLOTS):
                table[slot] = value
        pit = MS.PITCHERS.loadings.get(kind, {})
        if pit:
            table["P"] = float(pit.get("P", next(iter(pit.values()))))
        out[kind] = table
    return out


def board_spec():
    """A spec whose loadings are keyed the way the BOARD is keyed.

    The simulator looks a player's loadings up by his `position`, and by the
    time the pool reaches it that column holds DraftKings roster positions -
    OF and P - while the sport's own spec is keyed by box-score positions -
    LF, CF, RF, DH, SP, RP. Handing it the raw spec silently dropped every
    outfielder and every pitcher onto a default loading, which is to say the
    server's own two lineups were solved against a correlation structure the
    page's browser search does not share. The two are supposed to be
    comparable; that is the entire point of shipping both.
    """
    import dataclasses
    return dataclasses.replace(
        MS.HITTERS,
        positions=sorted(HITTER_SLOTS | {"P"}),
        loadings=loadings_for_page())


def server_lineups(pool: pd.DataFrame, draws: np.ndarray,
                   own: pd.Series, field_size: int) -> dict:
    """The exact integer program's answer, shipped alongside the browser's.

    The page says this is from the last scheduled run and solved exactly. It
    is the only thing on the page the browser's own near-optimal search can be
    checked against, so a failure here is reported and left empty rather than
    filled with the browser's kind of answer wearing the server's label.
    """
    out = {}
    for objective in ("cash", "gpp"):
        try:
            built = O.build(pool, ROSTER, draws, objective=objective,
                            entries=1, own=own, field_size=field_size)
        except Exception as exc:                           # noqa: BLE001
            log.error("the %s integer program did not solve (%s: %s)",
                      objective, type(exc).__name__, str(exc)[:120])
            continue
        if not len(built):
            continue
        out[objective] = {
            "players": [str(n) for n in built["name"]],
            "salary": int(pd.to_numeric(built.get("charged",
                                                  built["salary"])).sum()),
            "ceiling": round(float(pd.to_numeric(
                built["ceiling"], errors="coerce").sum()), 1),
        }
    return out


def candidate_slates(draft_group: int | None, look: int) -> list[tuple]:
    """Which boards to publish, biggest first.

    Ranking by contest COUNT - which is what the football build does - picked
    a three-game early slate over the main evening board, because cheap early
    contests are numerous. Baseball's useful slate is the one with the most
    games in it, and the only way to know how many games a draft group covers
    is to fetch it, so the top few by contest count are fetched and then
    re-sorted by how many teams they actually contain.

    Every one that survives is published. The page already has a slate picker;
    filling it is more useful than guessing which single board you wanted.
    """
    listed = MD.slates()
    if listed.empty:
        sys.exit("DraftKings is listing no baseball slates right now")

    if draft_group:
        row = listed[listed["draft_group"] == draft_group]
        label = str(row["example"].iloc[0]) if len(row) else "(given)"
        return [(int(draft_group), label, MD.board(int(draft_group)))]

    classic = listed[~listed["game_type"].astype(str)
                     .str.contains("showdown", case=False, na=False)]
    out = []
    for r in classic.head(look).itertuples(index=False):
        try:
            board = MD.board(int(r.draft_group))
        except Exception as exc:                               # noqa: BLE001
            log.info("draft group %s did not load (%s)", r.draft_group,
                     str(exc)[:70])
            continue
        teams = int(board["team"].nunique())
        if teams < 2:
            continue
        out.append((int(r.draft_group), str(r.example), board, teams))
    if not out:
        sys.exit("no baseball board could be loaded")
    out.sort(key=lambda t: -t[3])
    log.info("boards found: %s",
             ", ".join(f"{t[0]} ({t[3]} teams)" for t in out))
    return [(dg, label, board) for dg, label, board, _ in out]


def build_slate(proj: pd.DataFrame, dg: int, label: str,
                board: pd.DataFrame, probables: dict, field_size: int,
                sims: int) -> dict | None:
    log.info("draft group %s: %s", dg, label)

    board["slot"] = board["position"].map(roster_position)
    unknown = board[board["slot"].isna()]
    if len(unknown):
        log.info("%d priced players fill no slot (%s) - dropped",
                 len(unknown),
                 ", ".join(sorted(set(unknown["position"].astype(str)))[:6]))
    board = board[board["slot"].notna()].copy()

    gone = board["disabled"].fillna(False).astype(bool)
    if gone.any():
        log.info("%d players are flagged unavailable and are dropped",
                 int(gone.sum()))
        board = board[~gone].copy()

    # Only today's announced starters may fill a P slot.
    #
    # Without this the board prices every pitcher on every 26-man roster, the
    # ones not starting are cheap, and points per dollar - the statistic a
    # pitcher who throws no innings maximises - puts one of them in every
    # single lineup. That is not a subtle mis-ranking; it is the optimiser
    # working perfectly on a board that lied to it.
    ids = _id_text(board["mlb_id"])
    is_p = board["slot"] == "P"
    starting = ids.isin(set(probables))
    drop = is_p & ~starting
    log.info("pitchers: %d priced, %d are today's announced starters",
             int(is_p.sum()), int((is_p & starting).sum()))
    if int((is_p & starting).sum()) < 2:
        log.error("fewer than two announced starters are priced on draft "
                  "group %s - the probables are not matching this board, and "
                  "publishing it would put a pitcher who is not playing into "
                  "every lineup. Skipped.", dg)
        return None
    if drop.any():
        names = ", ".join(board.loc[drop, "name"].head(6))
        log.info("dropping %d pitchers who are not starting today (%s%s)",
                 int(drop.sum()), names, " ..." if int(drop.sum()) > 6 else "")
        board = board[~drop].copy()

    merged = join_board(board, proj)
    pool = merged[merged["player_id"].notna()].copy()
    pool = pool[pd.to_numeric(pool["salary"], errors="coerce").notna()]
    if pool.empty:
        log.error("nothing on draft group %s joined to the history", dg)
        return None

    have = float(pd.to_numeric(pool["salary"]).sum()
                 / pd.to_numeric(merged["salary"], errors="coerce").sum())
    log.info("%d of %d priced players projected, %.0f%% of slate salary",
             len(pool), len(merged), 100 * have)

    missing = [s for s in set(ROSTER["slots"])
               if not (pool["slot"] == s).any()]
    if missing:
        log.error("draft group %s has no projected player for %s - a legal "
                  "lineup does not exist, so it is skipped rather than "
                  "published as a board that cannot be built from", dg,
                  missing)
        return None

    # The engine wants its own column names.
    pool = pool.rename(columns={"slot": "position"})
    pool["position"] = pool["position"].astype(str)
    pool = pool.reset_index(drop=True)

    own = OWN.project(pool, ROSTER)
    lev = OWN.leverage(pool, own)
    pool["ownership"] = own
    pool["leverage"] = lev

    quantiles = list(MS.HITTERS.quantiles)
    draws = S.simulate(pool, quantiles, sims, spec=board_spec())

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "sport": SPORT,
        "site": "dk",
        "kind": "classic",
        "draft_group": dg,
        "label": label,
        "field_size": field_size,
        "quantiles": quantiles,
        "loadings": loadings_for_page(),
        "roster": ROSTER,
        "players": slate_players(pool, quantiles, own, lev),
        "server_lineups": server_lineups(pool, draws, own, field_size),
    }
    return payload


def write(payloads: list[dict]) -> None:
    """Every slate this run produced, plus a manifest listing exactly those.

    The manifest is rebuilt rather than appended to, and yesterday's files are
    deleted. A stale slate left in the dropdown is worse than a missing one:
    it loads, it looks current, and every salary in it is a day old.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    fresh = set()
    slates = []
    for payload in payloads:
        name = (f"{payload['sport']}_{payload['site']}_{payload['kind']}_"
                f"{payload['draft_group']}.json")
        path = DATA / name
        path.write_text(json.dumps(payload, separators=(",", ":")))
        fresh.add(name)
        log.info("wrote %s (%.0f KB, %d players)", path,
                 path.stat().st_size / 1024, len(payload["players"]))
        slates.append({"sport": payload["sport"], "site": payload["site"],
                       "kind": payload["kind"], "label": payload["label"],
                       "file": f"data/{name}"})

    for old in DATA.glob(f"{SPORT}_*.json"):
        if old.name not in fresh:
            old.unlink()
            log.info("removed stale slate %s", old.name)

    (DATA / "manifest.json").write_text(json.dumps({
        "updated_at": payloads[0]["generated_at"],
        "slates": slates,
    }, indent=1))
    log.info("manifest lists %d slate(s)", len(slates))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int, default=None)
    p.add_argument("--first-season", type=int, default=2025)
    p.add_argument("--field", type=int, default=100_000)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--slates", type=int, default=3,
                   help="how many boards to publish, biggest first")
    p.add_argument("--date", default=None,
                   help="slate date YYYY-MM-DD (default: today, US Eastern)")
    p.add_argument("--look", type=int, default=8,
                   help="how many draft groups to fetch before ranking them")
    args = p.parse_args(argv)

    seasons = list(range(args.first_season,
                         datetime.now(timezone.utc).year + 1))
    hist = C.load(SPORT, seasons)
    log.info("%d player-games of history", len(hist))
    proj = pd.concat([project_half(hist, "hitters"),
                      project_half(hist, "pitchers")], ignore_index=True)

    # The baseball day, in US Eastern - which is the day DraftKings and the
    # league both mean. A 21:00 UTC run is still the same evening in New York,
    # but a UTC date would already have rolled over for anything after 20:00
    # Eastern and would ask the league about tomorrow's probables.
    today = args.date or (datetime.now(timezone.utc)
                          .astimezone(ZoneInfo("America/New_York"))
                          .strftime("%Y-%m-%d"))
    probables = MD.probable_pitchers(today)
    if not probables:
        sys.exit(f"the league has announced no probable pitchers for {today}. "
                 f"Publishing without them puts a pitcher who is not playing "
                 f"into every lineup, so nothing is published and the page "
                 f"keeps what it had.")

    payloads = []
    for dg, label, board in candidate_slates(args.draft_group, args.look):
        if len(payloads) >= args.slates:
            break
        got = build_slate(proj, dg, label, board, probables,
                          args.field, args.sims)
        if got:
            payloads.append(got)

    if not payloads:
        sys.exit("no slate could be built - nothing was published, so the "
                 "page keeps whatever it had")
    write(payloads)

    print()
    print("=" * 70)
    for payload in payloads:
        print(f"{payload['label']}  -  {len(payload['players'])} players "
              f"(draft group {payload['draft_group']})")
        for objective, lu in payload["server_lineups"].items():
            print(f"  {objective:<5} ${lu['salary']:,}  "
                  f"ceiling {lu['ceiling']}")
            print(f"        {', '.join(lu['players'])}")
        if not payload["server_lineups"]:
            print("  NO server lineup solved - the page will show only the "
                  "browser's own search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
