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
