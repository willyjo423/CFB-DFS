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
    left["mlb_id"] = left["mlb_id"].astype(str)
    right = proj.copy()
    right["player_id"] = right["player_id"].astype(str)

    merged = left.merge(right.drop(columns=["team", "name"]),
                        left_on="mlb_id", right_on="player_id", how="left")
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
            "pos": str(r["slot"]),
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


def pick_slate(draft_group: int | None) -> tuple[int, str]:
    slates = MD.slates()
    if draft_group:
        row = slates[slates["draft_group"] == draft_group]
        label = str(row["example"].iloc[0]) if len(row) else "(given)"
        return int(draft_group), label
    if slates.empty:
        sys.exit("DraftKings is listing no baseball slates right now")
    row = slates.iloc[0]
    return int(row["draft_group"]), str(row["example"])


def run(draft_group: int | None, seasons: list[int],
        field_size: int, sims: int) -> dict:
    hist = C.load(SPORT, seasons)
    log.info("%d player-games of history", len(hist))

    proj = pd.concat([project_half(hist, "hitters"),
                      project_half(hist, "pitchers")], ignore_index=True)

    dg, label = pick_slate(draft_group)
    log.info("draft group %s: %s", dg, label)
    board = MD.board(dg)

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

    merged = join_board(board, proj)
    pool = merged[merged["player_id"].notna()].copy()
    pool = pool[pd.to_numeric(pool["salary"], errors="coerce").notna()]
    if pool.empty:
        sys.exit("nothing on this slate joined to the history")

    have = float(pd.to_numeric(pool["salary"]).sum()
                 / pd.to_numeric(merged["salary"], errors="coerce").sum())
    log.info("%d of %d priced players projected, %.0f%% of slate salary",
             len(pool), len(merged), 100 * have)

    missing = [s for s in set(ROSTER["slots"])
               if not (pool["slot"] == s).any()]
    if missing:
        sys.exit(f"no projected player can fill {missing} - a legal lineup "
                 f"does not exist and publishing this would put a page up "
                 f"that cannot build one")

    # The engine wants its own column names.
    pool = pool.rename(columns={"slot": "position"})
    pool["position"] = pool["position"].astype(str)
    pool = pool.reset_index(drop=True)

    own = OWN.project(pool, ROSTER)
    lev = OWN.leverage(pool, own)
    pool["ownership"] = own
    pool["leverage"] = lev

    quantiles = list(MS.HITTERS.quantiles)
    draws = S.simulate(pool, quantiles, sims, spec=MS.HITTERS)

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


def write(payload: dict) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    name = f"{payload['sport']}_{payload['site']}_{payload['kind']}_" \
           f"{payload['draft_group']}.json"
    path = DATA / name
    path.write_text(json.dumps(payload, separators=(",", ":")))
    log.info("wrote %s (%.0f KB, %d players)", path,
             path.stat().st_size / 1024, len(payload["players"]))

    manifest_path = DATA / "manifest.json"
    slates = []
    if manifest_path.exists():
        try:
            slates = json.loads(manifest_path.read_text()).get("slates", [])
        except json.JSONDecodeError:
            log.warning("the manifest was unreadable and is being rebuilt")
    entry = {"sport": payload["sport"], "site": payload["site"],
             "kind": payload["kind"], "label": payload["label"],
             "file": f"data/{name}"}
    slates = [s for s in slates if s.get("file") != entry["file"]]
    slates.insert(0, entry)

    # A slate whose file no longer exists is dropped rather than listed. The
    # page has no way to report a 404 except as an empty board.
    slates = [s for s in slates if (DOCS / s["file"]).exists()]
    manifest_path.write_text(json.dumps({
        "updated_at": payload["generated_at"],
        "slates": slates,
    }, indent=1))
    log.info("manifest lists %d slate(s)", len(slates))
    return path


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int, default=None)
    p.add_argument("--first-season", type=int, default=2025)
    p.add_argument("--field", type=int, default=100_000)
    p.add_argument("--sims", type=int, default=20_000)
    args = p.parse_args(argv)

    seasons = list(range(args.first_season,
                         datetime.now(timezone.utc).year + 1))
    payload = run(args.draft_group, seasons, args.field, args.sims)
    write(payload)

    print()
    print("=" * 70)
    print(f"{len(payload['players'])} players on draft group "
          f"{payload['draft_group']}")
    for objective, lu in payload["server_lineups"].items():
        print(f"  {objective:<5} ${lu['salary']:,}  ceiling {lu['ceiling']}")
        print(f"        {', '.join(lu['players'])}")
    if not payload["server_lineups"]:
        print("  NO server lineup solved - the page will show only the "
              "browser's own search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
