"""Grade a board we published against the contest that actually happened.

    python score_slates.py --board docs/data/mlb/dk-classic-153501.json \
                           --contest contest-standings-195744962.csv \
                           --me gdl23

Why this exists
---------------
Every measurement in this project so far grades a PROJECTION. The model grade
says the ranking beats a season average; the calibration report says the
quantiles cover what they claim. Both were true on a night we finished 95th of
206. The grade report says so itself, at the bottom, in the section called what
this does not tell you:

    Not return on investment. That needs historical DraftKings salaries and
    contest results, and DraftKings publishes neither.

DraftKings does publish them, once, to the people who entered: the contest
standings export carries every player's actual ownership and actual score, and
every entry's lineup and finish. That file is the only ground truth this
project has ever had about a BET rather than a projection. This reads it.

What it answers
---------------
1. Was the pool complete? Players the field rostered that our board did not
   contain are invisible to every other measurement we take - the board simply
   has no row for them, so nothing flags it. On the DET @ BUF showdown three
   were missing, one of them in the winning lineup.
2. Was the ownership model right? Scored the same way the ownership module is
   calibrated - KL divergence, mean error, rank correlation - so the numbers
   here and the numbers in `ownership.check_calibration` mean the same thing.
3. Were the projections right, on THIS slate, against what happened?
4. What did it take to win, against what we simulated it would take?
5. Where would our own lineups have finished?

What it deliberately does not do
--------------------------------
It does not fit anything. Reading a winning lineup and adjusting a projection
towards it is selecting on the outcome - one sample, drawn from the extreme
tail, of a process that is mostly variance. The honest uses of a finished
contest are ownership, the bar, and pool completeness, and those are what this
prints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
# A period becomes a SPACE, not nothing. DraftKings writes "St.Brown" with no
# space where other sources write "St. Brown"; deleting the period turns those
# into "stbrown" and "st brown", which do not match. Amon-Ra St. Brown was 35%
# owned on the one board we have, so this single rule is worth stating.
_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$")
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _SUFFIX.sub("", s).strip()
    return s


# --------------------------------------------------------------------------
# The contest export
# --------------------------------------------------------------------------
# Layout, which DraftKings does not document: a left block of entries and a
# right block of players, side by side in the same rows, separated by a blank
# column. The two blocks have different lengths, so every row has to be read
# for both and neither can be assumed to end where the other does.
#
#   Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
def read_contest(path: Path) -> tuple[list[dict], dict]:
    entries: list[dict] = []
    players: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) >= 6 and row[0].strip().isdigit():
                try:
                    pts = float(row[4])
                except (TypeError, ValueError):
                    continue
                entries.append({"rank": int(row[0]), "name": row[2].strip(),
                                "points": pts, "lineup": row[5].strip()})
            if len(row) >= 11 and row[7].strip() and row[7].strip() != "Player":
                who = row[7].strip()
                slot = row[8].strip()
                try:
                    pct = float(row[9].strip().rstrip("%")) / 100.0
                    fpts = float(row[10])
                except (TypeError, ValueError):
                    continue
                p = players.setdefault(who, {"name": who, "own": 0.0,
                                             "by_slot": {}, "fpts": {}})
                # A showdown player appears twice, once as CPT and once as
                # FLEX, and the two are different rosterings of one man. The
                # board's single `own` number covers all six slots, so the
                # comparable quantity is the SUM.
                p["own"] += pct
                p["by_slot"][slot] = pct
                p["fpts"][slot] = fpts
    return entries, players


def base_points(p: dict) -> float:
    """The un-multiplied score. A captain's FPTS already has 1.5x in it."""
    if "FLEX" in p["fpts"]:
        return p["fpts"]["FLEX"]
    if "UTIL" in p["fpts"]:
        return p["fpts"]["UTIL"]
    if "CPT" in p["fpts"]:
        return p["fpts"]["CPT"] / 1.5
    return next(iter(p["fpts"].values()), 0.0)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r

    ra, rb = rank(a), rank(b)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def kl(model: list[float], actual: list[float]) -> float:
    ms, as_ = sum(model), sum(actual)
    if ms <= 0 or as_ <= 0:
        return float("nan")
    out = 0.0
    for m, a in zip(model, actual):
        q = a / as_
        if q > 0:
            out += q * math.log(q / max(m / ms, 1e-9))
    return out


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report(board: dict, entries: list[dict], actual: dict,
           me: str | None) -> str:
    L: list[str] = []
    W = 74

    def head(title):
        L.append("")
        L.append(title)
        L.append("-" * W)

    modelled = {}
    for p in board.get("players", []):
        key = norm(p.get("name"))
        # A duplicated row is itself a bug, and summing is the honest read of
        # what the published board would have the field do.
        if key in modelled:
            modelled[key]["own"] += float(p.get("own") or 0.0)
            modelled[key]["dupe"] += 1
            continue
        modelled[key] = {"name": p.get("name"), "pos": p.get("pos"),
                         "salary": p.get("salary"),
                         "own": float(p.get("own") or 0.0),
                         "med": float(p.get("med") or 0.0),
                         "status": p.get("status", ""), "dupe": 1}

    act = {norm(k): v for k, v in actual.items()}
    both = [k for k in modelled if k in act]

    L.append("=" * W)
    L.append(f"{board.get('game_type', '?')}  draft group "
             f"{board.get('draft_group', '?')}  ({board.get('sport', '?')})")
    L.append(f"generated {board.get('generated_at', '?')}")
    L.append(f"{len(entries)} entries, {len(actual)} priced players, "
             f"{len(modelled)} on our board, {len(both)} joined")
    L.append("=" * W)

    # ---------------------------------------------------------------- pool
    head("1. WAS THE POOL COMPLETE?")
    missing = sorted((k for k in act if k not in modelled),
                     key=lambda k: -act[k]["own"])
    real = [k for k in missing if act[k]["own"] >= 0.005]
    if not real:
        L.append("  every player the field rostered was on our board.")
    else:
        L.append(f"  {len(real)} players the field rostered were NOT on our "
                 f"board at all.")
        L.append("  Nothing else we measure can see these - there is no row "
                 "to be wrong about.")
        L.append("")
        L.append(f"  {'player':<26}{'owned':>8}{'scored':>9}")
        for k in real[:15]:
            a = act[k]
            L.append(f"  {a['name']:<26}{pct(a['own']):>8}"
                     f"{base_points(a):9.1f}")
        top = entries[0]["lineup"] if entries else ""
        hit = [act[k]["name"] for k in real if norm(act[k]["name"]) in
               {norm(x) for x in re.split(r"\s{2,}|(?<=[a-z])\s(?=[A-Z]{2,})",
                                          top)}]
        inwin = [act[k]["name"] for k in real
                 if act[k]["name"].lower() in top.lower()]
        if inwin:
            L.append("")
            L.append(f"  IN THE WINNING LINEUP: {', '.join(inwin)}")
            L.append("  The winning lineup was not buildable from our board.")

    dupes = [m for m in modelled.values() if m["dupe"] > 1]
    if dupes:
        L.append("")
        L.append(f"  {len(dupes)} players appear MORE THAN ONCE on our board "
                 f"(a join bug, not a modelling one):")
        for d in dupes[:10]:
            L.append(f"    {d['name']} x{d['dupe']}")

    # ----------------------------------------------------------- ownership
    head("2. WAS THE OWNERSHIP MODEL RIGHT?")
    if len(both) < 5:
        L.append("  too few joined players to score.")
    else:
        m = [modelled[k]["own"] for k in both]
        a = [act[k]["own"] for k in both]
        mae = sum(abs(x - y) for x, y in zip(m, a)) / len(both)
        bias = sum(x - y for x, y in zip(m, a)) / len(both)
        L.append(f"  KL(actual||model) {kl(m, a):.3f}     "
                 f"mean abs error {100 * mae:.1f}pp     "
                 f"bias {100 * bias:+.1f}pp")
        L.append(f"  spearman rho      {spearman(m, a):.2f}     "
                 f"model sums to {100 * sum(m):.0f}%, "
                 f"actual to {100 * sum(a):.0f}%")
        sm, sa = sorted(m, reverse=True), sorted(a, reverse=True)
        n8 = min(8, len(both))
        L.append(f"  share on the top {n8}:  model {pct(sum(sm[:n8]) / sum(sm))}"
                 f"   actual {pct(sum(sa[:n8]) / sum(sa))}")
        L.append("")
        L.append("  worst misses:")
        L.append(f"  {'player':<26}{'model':>8}{'actual':>8}{'err':>8}"
                 f"{'scored':>9}")
        for k in sorted(both,
                        key=lambda k: -abs(modelled[k]["own"] - act[k]["own"])
                        )[:12]:
            e = modelled[k]["own"] - act[k]["own"]
            L.append(f"  {modelled[k]['name']:<26}"
                     f"{pct(modelled[k]['own']):>8}{pct(act[k]['own']):>8}"
                     f"{100 * e:+7.1f}p{base_points(act[k]):9.1f}")

    # ---------------------------------------------------------- projection
    head("3. WERE THE PROJECTIONS RIGHT?")
    proj = [k for k in both if modelled[k]["med"] > 0 or act[k]["fpts"]]
    if len(proj) < 5:
        L.append("  too few joined players to score.")
    else:
        errs = [(k, base_points(act[k]) - modelled[k]["med"]) for k in proj]
        mae = sum(abs(e) for _, e in errs) / len(errs)
        bias = sum(e for _, e in errs) / len(errs)
        L.append(f"  mean abs error {mae:.2f} points     "
                 f"bias {bias:+.2f} (actual minus our median)")
        L.append(f"  rank correlation with what actually happened: "
                 f"{spearman([modelled[k]['med'] for k in proj], [base_points(act[k]) for k in proj]):.2f}")
        zeros = [k for k in proj if base_points(act[k]) <= 0]
        L.append(f"  {len(zeros)} of {len(proj)} joined players scored zero "
                 f"or less.")
        worst = sorted(zeros, key=lambda k: -modelled[k]["own"])[:6]
        if worst:
            L.append("  the ones we told you to own anyway:")
            for k in worst:
                L.append(f"    {modelled[k]['name']:<24}"
                         f"{pct(modelled[k]['own'])} owned by us, "
                         f"projected {modelled[k]['med']:.1f}, scored "
                         f"{base_points(act[k]):.1f}"
                         + (f"   [{modelled[k]['status']}]"
                            if modelled[k]["status"] not in ("", "clear")
                            else ""))

    # ----------------------------------------------------------- the bar
    head("4. WHAT DID IT TAKE TO WIN?")
    if entries:
        pts = sorted((e["points"] for e in entries), reverse=True)
        n = len(pts)

        def at(q):
            return pts[min(n - 1, max(0, int(round((1 - q) * n)) - 1))]

        L.append(f"  won with {pts[0]:.1f}     "
                 f"top 1% {at(0.99):.1f}     top 10% {at(0.90):.1f}     "
                 f"median {at(0.50):.1f}")
        sim = board.get("server_lineups") or []
        if sim:
            best = max((float(s.get("median") or s.get("proj") or 0.0)
                        for s in sim), default=0.0)
            ceil = max((float(s.get("ceiling") or 0.0) for s in sim),
                       default=0.0)
            if best:
                L.append(f"  our best server lineup projected {best:.1f} "
                         f"median, {ceil:.1f} ceiling")
                L.append(f"  the winner scored {pts[0] / best:.2f}x our median "
                         f"projection"
                         + (f", {pts[0] / ceil:.2f}x our ceiling"
                            if ceil else ""))
                if ceil and pts[0] > ceil:
                    L.append("  -> the winning score was ABOVE the top of our "
                             "simulated range. The")
                    L.append("     simulator's tail is too thin for this "
                             "contest size.")

    # -------------------------------------------------------------- us
    if me:
        head(f"5. WHERE DID {me.upper()} FINISH?")
        mine = [e for e in entries if e["name"].split(" (")[0].strip().lower()
                == me.strip().lower()]
        if not mine:
            L.append(f"  no entry named {me} in this contest.")
        else:
            n = len(entries)
            for e in sorted(mine, key=lambda e: e["rank"]):
                L.append(f"  rank {e['rank']} of {n}  ({e['points']:.1f} pts, "
                         f"beat {100 * (n - e['rank']) / n:.0f}% of the field)")
                L.append(f"    {e['lineup']}")

    L.append("")
    L.append("=" * W)
    L.append("Nothing here is fitted. A winning lineup is one sample from the")
    L.append("extreme tail; training a projection towards it is selecting on")
    L.append("the outcome. Ownership, the bar, and pool completeness are the")
    L.append("three things a finished contest can honestly tell you.")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", required=True,
                   help="a published slate JSON from docs/data/<sport>/, or "
                        "just the draft group number - the board is then "
                        "found by searching docs/data/")
    p.add_argument("--contest", required=True,
                   help="the DraftKings contest-standings CSV export")
    p.add_argument("--me", default=None,
                   help="your DraftKings username, to locate your entries")
    p.add_argument("--out", default=None, help="also write the report here")
    args = p.parse_args(argv)

    # A draft group is what the page and the log both show, so accept it in
    # place of a path. Typing an exact path into a workflow input is how a
    # useful tool goes unused.
    board_path = Path(args.board)
    if not board_path.exists():
        if args.board.strip().isdigit():
            hits = sorted(Path("docs/data").rglob(f"*-{args.board.strip()}.json"))
            if not hits:
                have = sorted(p.name for p in Path("docs/data").rglob("*.json")
                              if p.name != "manifest.json")
                print(f"No board for draft group {args.board}. "
                      f"docs/data holds:\n  " + "\n  ".join(have or ["nothing"]),
                      file=sys.stderr)
                return 1
            board_path = hits[0]
            print(f"board: {board_path}")
        else:
            print(f"No such board file: {args.board}", file=sys.stderr)
            return 1
    board = json.loads(board_path.read_text())
    entries, actual = read_contest(Path(args.contest))
    if not actual:
        print("No player rows found in that CSV. The export needs the "
              "right-hand block with Player / Roster Position / %Drafted.",
              file=sys.stderr)
        return 1

    text = report(board, entries, actual, args.me)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
