"""Generate and verify independently seeded, D4-disjoint PMUT layout collections."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
COUNTS = (6, 12, 24, 50, 150, 400, 800, 1250, 1700, 2100, 2300, 2460, 2488, 2494)
OFF_GRID = (9, 18, 90, 625, 1950, 2491)
FAMILIES = ("random", "cluster", "edge", "stripe", "patch")
SPLITS = {
    "references": ("R", 45, 284001, COUNTS),
    "calibration": ("C", 39, 284002, COUNTS),
    "test": ("T", 30, 284003, COUNTS),
    "offgrid": ("G", 12, 284004, OFF_GRID),
}
PRIOR = ("S074/masks.npz", "S194/new_masks.npz", "S197/predictions.npz",
         "S197/boundary_predictions.npz")
Y, X = np.indices((50, 50))
EDGE = np.minimum.reduce([Y, X, 49 - Y, 49 - X])


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def orbit(mask):
    """Canonical packed row-major bytes under all eight symmetries."""
    return min(np.packbits(np.rot90(view, k)).tobytes()
               for view in (mask, mask[:, ::-1]) for k in range(4))


def old_orbits():
    seen = set()
    inventory = {}
    for name in PRIOR:
        path = ROOT.parent / name
        with np.load(path) as data:
            if "packed" in data:
                masks = np.unpackbits(data["packed"], axis=1, count=2500).reshape(-1, 50, 50)
            else:
                masks = data["mask"]
        seen.update(orbit(mask) for mask in masks)
        inventory[name] = {"rows": len(masks), "sha256": file_hash(path)}
    return seen, inventory


def smooth_noise(noise, sigma):
    # Half-sample symmetric reflection, separable convolution, four-sigma truncation.
    radius = int(4 * sigma + 0.5)
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    out = noise
    for axis, padding in enumerate([((radius, radius), (0, 0)), ((0, 0), (radius, radius))]):
        padded = np.pad(out, padding, mode="symmetric")
        windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
        out = np.einsum("ijk,k->ij", windows, kernel)
    return out


def make_mask(rng, count, family):
    noise = rng.standard_normal((50, 50))
    params = {}
    if family == "random":
        scores = noise
    elif family == "cluster":
        cy, cx = rng.uniform(4, 46, 2)
        sigma = rng.uniform(0.4, 3)
        scores = -np.hypot(Y - cy, X - cx) + sigma * noise
        params = dict(cy=cy, cx=cx, noise_scale=sigma)
    elif family == "edge":
        sigma = rng.uniform(0.6, 3)
        scores = -EDGE + sigma * noise
        params = dict(noise_scale=sigma)
    elif family == "stripe":
        theta, omega = rng.uniform(0, np.pi), rng.uniform(0.3, 1.1)
        phase, sigma = rng.uniform(0, 2 * np.pi), rng.uniform(0.16, 0.18)
        scores = np.sin((X * np.cos(theta) + Y * np.sin(theta)) * omega + phase) + sigma * noise
        params = dict(theta_rad=theta, omega_rad_per_cell=omega, phase_rad=phase, noise_scale=sigma)
    elif family == "patch":
        sigma = rng.uniform(1.4, 4.0)
        scores = smooth_noise(noise, sigma)
        params = dict(sigma_cells=sigma)
    else:
        raise ValueError(family)
    active = np.argpartition(scores.ravel(), -count)[-count:]
    mask = np.zeros(2500, dtype=np.uint8)
    mask[active] = 1
    return mask.reshape(50, 50), params


def save_split(name, seen):
    prefix, reps, seed, levels = SPLITS[name]
    rng = np.random.Generator(np.random.PCG64(seed))
    masks, labels, metadata = [], [], []
    rejected = 0
    for k in levels:
        for label in FAMILIES:
            for _ in range(reps):
                while True:
                    mask, params = make_mask(rng, k, label)
                    canonical = orbit(mask)
                    if canonical not in seen:
                        break
                    rejected += 1
                seen.add(canonical)
                masks.append(mask)
                labels.append(label)
                metadata.append(dict(split=name, k=k, label=label, params=params,
                                     mask_sha256=sha256(mask.tobytes()).hexdigest(),
                                     orbit_sha256=sha256(canonical).hexdigest()))
    order = rng.permutation(len(masks))
    ids = np.array([f"{prefix}{n:04d}" for n in range(1, len(masks) + 1)])
    pack = ROOT / f"{name}.npz"
    with pack.with_suffix(".npz.part").open("wb") as out:
        np.savez_compressed(out, masks=np.stack(masks)[order], labels=np.array(labels)[order], ids=ids)
    pack.with_suffix(".npz.part").replace(pack)
    rows = [dict(id=str(id_), **metadata[j]) for id_, j in zip(ids, order)]
    return rows, rejected


def generate():
    seal = ROOT / "bank_manifest.json"
    if seal.exists():
        raise FileExistsError("Bank already sealed. Run bank.py --verify to audit the saved files.")
    seen, prior = old_orbits()
    prior_count = len(seen)
    rows, duplicates = [], {}
    for split in SPLITS:
        split_rows, duplicates[split] = save_split(split, seen)
        rows.extend(split_rows)
    with (ROOT / "bank_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "split", "k", "label", "params",
                                                    "mask_sha256", "orbit_sha256"))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, params=json.dumps(row["params"], sort_keys=True)))
    files = ["bank.py", "bank_notes.md", "bank_inventory.csv", *[f"{s}.npz" for s in SPLITS]]
    manifest = dict(archive=prior, archive_unique_orbits=prior_count, counts=COUNTS, families=FAMILIES,
                    splits={s: dict(seed=seed, levels=levels, per_cell=reps, count=reps * len(levels) * len(FAMILIES),
                                    duplicate_candidates_rejected=duplicates[s]) for s, (_, reps, seed, levels) in SPLITS.items()},
                    bit_generator="PCG64", numpy_version=np.__version__, python_version=sys.version.split()[0],
                    files={name: dict(bytes=(ROOT / name).stat().st_size, sha256=file_hash(ROOT / name))
                           for name in files})
    part = seal.with_suffix(".json.part")
    part.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    part.replace(seal)


def verify():
    seal = json.loads((ROOT / "bank_manifest.json").read_text(encoding="utf-8"))
    seen, prior = old_orbits()
    assert prior == seal["archive"] and len(seen) == seal["archive_unique_orbits"]
    for name, record in seal["files"].items():
        path = ROOT / name
        assert path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"]
    with (ROOT / "bank_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    by_id = {r["id"]: r for r in inventory}
    assert len(by_id) == len(inventory)
    new_ids = set()
    for name, (prefix, reps, _, levels) in SPLITS.items():
        cells = Counter()
        with np.load(ROOT / f"{name}.npz", allow_pickle=False) as data:
            masks, ids, labels = data["masks"], data["ids"], data["labels"]
        assert masks.dtype == np.uint8 and masks.shape == (len(levels) * len(FAMILIES) * reps, 50, 50)
        assert len(ids) == len(labels) == len(masks) and not new_ids.intersection(ids)
        assert len(set(ids)) == len(ids) and all(id_.startswith(prefix) for id_ in ids)
        assert (masks <= 1).all()
        new_ids.update(ids)
        for id_, label, mask in zip(ids, labels, masks):
            k, key, row = int(mask.sum()), orbit(mask), by_id[id_]
            assert key not in seen
            seen.add(key)
            assert row["split"] == name and row["label"] == label and int(row["k"]) == k
            assert sha256(mask.tobytes()).hexdigest() == row["mask_sha256"]
            assert sha256(key).hexdigest() == row["orbit_sha256"]
            cells[k, label] += 1
        assert cells == {(k, lab): reps for k in levels for lab in FAMILIES}
        print(f"{name}: {len(masks)} rows, {len(cells)} strata, D4 and hashes verified.")
    assert new_ids == by_id.keys()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="read and verify an existing bank")
    if not parser.parse_args().verify:
        generate()
    verify()
