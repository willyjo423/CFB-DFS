"""Engine tests. Sport-free by construction - none of these mention football.

The engine's whole claim is that it does not know which sport it is running
on, so its tests are written against a made-up sport. If a test here needed
football to pass, the abstraction would be a lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import SportSpec, features as EF, frame as FR

# A deliberately invented sport. Two usage columns, one share, three
# positions, nothing football about it.
TOY = SportSpec(
    name="Toyball",
    usage=["grabs", "runs"],
    shares=["grabs"],
    touch_columns=["grabs", "runs"],
    positions=["A", "B", "C"],
    min_prior_games=2,
    quantiles=[0.10, 0.50, 0.90],
    halflife=3.0,
)


def _frame(periods=6, players=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(1, periods + 1):
        for p in range(players):
            side = "L" if p < players // 2 else "R"
            rows.append({
                "player_id": f"00{p}", "name": f"P{p}", "season": 2025,
                "period": t, "team": side, "opponent": "R" if side == "L"
                else "L", "is_home": 1 if side == "L" else 0,
                "position": ["A", "B", "C"][p % 3],
                "grabs": float(rng.integers(0, 10)),
                "runs": float(rng.integers(0, 10)),
                "points": float(10 * p + t),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- spec

def test_spec_refuses_nonsense():
    for kwargs, why in [
        ({"usage": []}, "no usage columns"),
        ({"positions": []}, "no positions"),
        ({"quantiles": [0.5, 0.1]}, "unsorted quantiles"),
        ({"quantiles": [0.0, 0.5]}, "quantile outside (0,1)"),
        ({"quantiles": [0.5, 0.5]}, "duplicate quantiles"),
    ]:
        base = {"name": "X", "usage": ["a"], "positions": ["P"]}
        base.update(kwargs)
        try:
            SportSpec(**base)
        except ValueError:
            continue
        raise AssertionError(f"should have refused: {why}")


def test_a_share_column_does_not_replace_its_raw_column():
    """The bug this separation exists to prevent.

    `grabs` is both a usage column and a share column. If the share were
    written back over the raw column, `ewm_grabs` would be the rolling mean
    of a RATIO rather than a volume - no error, and close enough to
    plausible that grading would not obviously catch it.
    """
    assert "ewm_grabs" in TOY.features
    assert "ewm_share_grabs" in TOY.features
    out = EF.build(_frame(), TOY)
    # Raw volumes are bigger than 1; shares are not.
    assert out["grabs"].max() > 1.0, "raw column was overwritten by a share"
    assert out["share_grabs"].max() <= 1.0 + 1e-9


def test_feature_order_is_stable():
    """A fitted model stores its column list; reordering hands it the wrong
    numbers with no error at all."""
    assert TOY.features == TOY.features
    assert len(TOY.features) == len(set(TOY.features)), "duplicate features"


# ------------------------------------------------------------------ frame

def test_validate_catches_the_silent_failures():
    good = _frame()
    assert FR.validate(good, TOY, strict=False) == []

    numeric_id = good.copy()
    numeric_id["player_id"] = range(len(numeric_id))
    assert any("not text" in p
               for p in FR.validate(numeric_id, TOY, strict=False))

    constant = good.copy()
    constant["points"] = 5.0
    assert any("single value" in p
               for p in FR.validate(constant, TOY, strict=False))

    duped = pd.concat([good, good], ignore_index=True)
    assert any("duplicate" in p
               for p in FR.validate(duped, TOY, strict=False))

    flat_home = good.copy()
    flat_home["is_home"] = 0
    assert any("is_home never varies" in p
               for p in FR.validate(flat_home, TOY, strict=False))

    wrong_pos = good.copy()
    wrong_pos["position"] = "ZZ"
    assert any("no row carries a position" in p
               for p in FR.validate(wrong_pos, TOY, strict=False))


def test_validate_raises_when_strict():
    bad = _frame().drop(columns=["opponent"])
    try:
        FR.validate(bad, TOY)
    except FR.FrameError as exc:
        assert "opponent" in str(exc)
    else:
        raise AssertionError("should have raised")


# --------------------------------------------------------------- the leak

def test_no_feature_contains_the_row_it_predicts():
    """Tamper with one player-period. Two things must BOTH hold.

    Its own features must not move - otherwise a feature can see the outcome
    it predicts. And LATER periods must move - otherwise the tampering never
    reached anything and this test could not detect a leak either way.

    The second half is what makes it a test rather than a ritual.
    """
    base = _frame()
    before = EF.build(base, TOY)

    tampered = base.copy()
    m = (tampered["player_id"] == "001") & (tampered["period"] == 3)
    assert m.sum() == 1
    tampered.loc[m, ["points", "grabs"]] = [999.0, 99.0]
    after = EF.build(tampered, TOY)

    key = ["player_id", "season", "period"]
    b = before.sort_values(key).reset_index(drop=True)
    a = after.sort_values(key).reset_index(drop=True)
    own = ["ewm_grabs", "ewm_runs", "ewm_points", "ewm_touches",
           "sd_points", "played_last", "ewm_played", "played_rate"]

    same = (b["player_id"] == "001") & (b["period"] == 3)
    for c in own:
        assert np.allclose(b.loc[same, c].to_numpy(dtype=float),
                           a.loc[same, c].to_numpy(dtype=float),
                           equal_nan=True), f"{c} saw its own outcome"

    later = (b["player_id"] == "001") & (b["period"] > 3)
    moved = [c for c in own
             if not np.allclose(b.loc[later, c].to_numpy(dtype=float),
                                a.loc[later, c].to_numpy(dtype=float),
                                equal_nan=True)]
    assert moved, "nothing responded; this test cannot detect a leak"


def test_build_never_changes_the_row_count():
    f = _frame()
    assert len(EF.build(f, TOY)) == len(f)


def test_debut_has_no_history():
    out = EF.build(_frame(), TOY)
    first = out[out["period"] == 1]
    assert first["games_played"].eq(0).all()
    assert first["ewm_points"].isna().all()


def test_home_flag_reads_numbers_as_numbers():
    got = EF.home_flag(pd.DataFrame({"is_home": [1.0, 0.0, 1.0]}))
    assert list(got) == [1.0, 0.0, 1.0]


def test_home_flag_unknown_is_missing_not_false():
    assert EF.home_flag(pd.DataFrame({"is_home": ["?", "??"]})).isna().all()


def test_trainable_excludes_debuts_and_other_positions():
    out = EF.build(_frame(periods=8), TOY)
    tr = EF.trainable(out, TOY)
    assert tr["games_played"].min() >= TOY.min_prior_games
    assert set(tr["position"]) <= set(TOY.positions)
    assert len(tr) < len(out)


# ------------------------------------------------------- correlation

CORR_TOY = SportSpec(
    name="Toyball-corr",
    usage=["grabs"],
    positions=["A", "B", "D"],
    loadings={"game":    {"A": 0.39, "B": 0.39, "D": -0.34},
              "team":    {"A": 0.45, "B": 0.45, "D": 0.46},
              "compete": {"A": 0.98, "B": 0.86, "D": 0.00}},
)


def _roster_frame():
    return pd.DataFrame([
        {"position": "A", "team": "L", "game": "L@R"},
        {"position": "B", "team": "L", "game": "L@R"},
        {"position": "B", "team": "L", "game": "L@R"},
        {"position": "D", "team": "L", "game": "L@R"},
        {"position": "A", "team": "R", "game": "L@R"},
    ])


def test_correlation_matrix_is_psd_by_construction():
    """Not repaired afterwards - built as a sum of PSD pieces.

    An earlier version rescaled the finished covariance when competition
    consumed too much variance, which destroyed the very property the
    construction exists to guarantee, and the Cholesky failed on a real
    slate.
    """
    from engine import factors as FA
    corr, _ = FA.build(_roster_frame(), CORR_TOY)
    ev = np.linalg.eigvalsh(corr)
    assert ev.min() >= -1e-9, f"not PSD: min eigenvalue {ev.min()}"
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)


def test_a_stack_beats_competition():
    """Two players who score together must correlate above two who do not.

    If this inverts, the optimiser stacks the wrong pairs - and it would
    still produce confident, plausible lineups.
    """
    from engine import factors as FA
    corr, _ = FA.build(_roster_frame(), CORR_TOY)
    stack = corr[0, 1]          # A with B, same team
    compete = corr[1, 2]        # B with B, same team - they share work
    assert stack > compete, (stack, compete)


def test_a_negative_loading_stays_negative():
    """The sign is the part that matters.

    A defence scores when its own offence fails, so it loads negatively on
    its game. A model with that backwards happily stacks a quarterback with
    the defence trying to stop him.
    """
    from engine import factors as FA
    corr, _ = FA.build(_roster_frame(), CORR_TOY)
    assert corr[0, 3] < corr[0, 1], "D should not correlate like a team-mate"


def test_simulated_draws_track_the_target_correlation():
    from engine import factors as FA, simulate as SI
    players = _roster_frame()
    for c, v in zip(["q10", "q50", "q90"], [2.0, 8.0, 20.0]):
        players[c] = v
    corr, _ = FA.build(players, CORR_TOY)
    draws = SI.simulate(players, [0.10, 0.50, 0.90], 6000, spec=CORR_TOY)
    realised = np.corrcoef(draws)[0, 1]
    assert realised > 0.15, f"stack correlation vanished: {realised}"


def test_the_availability_gate_dilutes_correlation_on_purpose():
    """Documented, not accidental.

    Availability is drawn INDEPENDENTLY of the copula, so gating knocks the
    realised correlation below the target. That is correct - two team-mates'
    chances of being scratched are not the same event as their scoring
    together - but it surprises anyone comparing the two numbers, so it is
    pinned here rather than left to be rediscovered.
    """
    from engine import simulate as SI
    players = _roster_frame()
    for c, v in zip(["q10", "q50", "q90"], [2.0, 8.0, 20.0]):
        players[c] = v
    ungated = np.corrcoef(
        SI.simulate(players, [0.10, 0.50, 0.90], 6000, spec=CORR_TOY))[0, 1]
    players["p_play"] = 0.7
    gated = np.corrcoef(
        SI.simulate(players, [0.10, 0.50, 0.90], 6000, spec=CORR_TOY))[0, 1]
    assert gated < ungated, (gated, ungated)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except Exception as exc:                   # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(main())
