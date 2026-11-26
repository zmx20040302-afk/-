"""Replay a geometry-only family comparison on the saved S194 and S197 masks."""
import csv
import hashlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FAMILIES = ("random", "cluster", "edge", "stripe", "patch")
FEATURES = ("edge_shift_cells", "gyration_cells", "interface_ratio",
            "window_variance", "periodic_peak_fraction", "low_frequency_fraction")
Y, X = np.indices((50, 50))
EDGE = np.minimum.reduce([Y, X, 49 - Y, 49 - X])
WAVE = np.fft.fftfreq(50) * 50
RADIUS = np.hypot(WAVE[:, None], WAVE[None, :])
PERIODIC = (RADIUS >= 2) & (RADIUS <= 12)
LOW = (RADIUS > 0) & (RADIUS <= 2)


def features(mask):
    n = mask.size
    k = int(mask.sum())
    if k == 0 or k == n:
        raise ValueError("A uniform mask has no identifiable family.")
    on = mask.astype(bool)
    y, x = Y[on], X[on]
    edge_shift = EDGE.mean() - EDGE[on].mean()
    gyration = np.sqrt(y.var() + x.var())
    different = (np.count_nonzero(on[1:] != on[:-1]) +
                 np.count_nonzero(on[:, 1:] != on[:, :-1]))
    random_expected = 4900 * 2 * k * (n - k) / (n * (n - 1))
    cumulative = np.pad(on.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = (cumulative[5:, 5:] - cumulative[:-5, 5:] -
              cumulative[5:, :-5] + cumulative[:-5, :-5])
    power = np.abs(np.fft.fft2(on - k / n)) ** 2
    power[0, 0] = 0
    total = power.sum()
    return np.array([edge_shift, gyration, different / random_expected,
                     (counts / 25).var(), 2 * power[PERIODIC].max() / total,
                     power[LOW].sum() / total])


def orbit(mask):
    return min(np.packbits(np.rot90(view, rotation)).tobytes()
               for view in (mask, mask[:, ::-1]) for rotation in range(4))


def write_rows(filename, rows):
    with (HERE / filename).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with np.load(BASE / "S194/new_masks.npz") as source:
        train_masks = np.unpackbits(source["packed"], axis=1, count=2500).reshape(-1, 50, 50)
        aliases = {"stripes": "stripe", "patches": "patch"}
        labels = np.array([aliases.get(s, s) for s in source["family"]])
    with (BASE / "S197/cases.csv").open(encoding="utf-8-sig") as handle:
        meta = list(csv.DictReader(handle))
    with np.load(BASE / "S197/predictions.npz") as source:
        all_test = source["mask"]
    selected = [i for i, row in enumerate(meta) if row["cohort"] == "stratified"]
    assert len(train_masks) == 2500 and len(selected) == 135
    train_k = train_masks.sum(axis=(1, 2)).astype(int)
    levels = np.unique(train_k)
    assert all(np.count_nonzero((train_k == k) & (labels == f)) == 10
               for k in levels for f in FAMILIES)
    training_orbits = {orbit(m) for m in train_masks}
    test_orbits = {orbit(all_test[i]) for i in selected}
    assert len(test_orbits) == 135 and training_orbits.isdisjoint(test_orbits)
    training = np.stack([features(mask) for mask in train_masks])
    assert np.isfinite(training).all()
    rows = []
    for index in selected:
        mask, row = all_test[index], meta[index]
        assert hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest() == row["sha256"]
        k = int(mask.sum())
        assert k == int(row["active"])
        value = features(mask)
        for view in (mask, mask[:, ::-1]):
            for rotation in range(4):
                np.testing.assert_allclose(value, features(np.rot90(view, rotation)), atol=1e-12)
        local_levels = sorted(levels, key=lambda level: (abs(level - k), level))[:2]
        use = np.isin(train_k, local_levels)
        scale = training[use].std(axis=0)
        assert np.all(scale > 0)
        distances = np.linalg.norm((training[use] - value) / scale, axis=1)
        scores = {f: np.sort(distances[labels[use] == f])[:3].mean() for f in FAMILIES}
        best, second = sorted(scores, key=scores.get)[:2]
        rows.append(dict(id=row["id"], k=k, generator=row["shape"], assigned=best,
                         runner_up=second, calibration_k1=local_levels[0],
                         calibration_k2=local_levels[1], **dict(zip(FEATURES, value)),
                         **{"distance_" + f: v for f, v in scores.items()}))
    write_rows("assignments.csv", rows)
    print("expected/predicted", *FAMILIES)
    for family in FAMILIES:
        print(family, *[sum(r["generator"] == family and r["assigned"] == f for r in rows)
                        for f in FAMILIES])
    print("matched by K", [(k, sum(r["assigned"] == r["generator"] for r in rows if r["k"] == k))
                            for k in sorted({r["k"] for r in rows})])
    for r in rows:
        if r["generator"] != r["assigned"]:
            print("mismatch", r["id"], r["k"], r["generator"], r["assigned"])


if __name__ == "__main__":
    main()
