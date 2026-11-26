"""Propose fixed-count layouts by fitting the geometry to the supplied pressure field."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from bank import EDGE, FAMILIES, X, Y, smooth_noise
from common import roi


@lru_cache(maxsize=1)
def receiver_rows():
    with np.load(Path(__file__).with_name("geometry_operator.npz")) as data:
        transfer = data["transfer"].reshape(-1, 2500).T
        rows = np.ascontiguousarray(np.concatenate([transfer.real, transfer.imag], axis=1))
        return rows, data["indices"], data["grid"]


@lru_cache(maxsize=1)
def sparse_dictionary():
    rows, _, _ = receiver_rows()
    size = rows.shape[1] // 2
    energy = rows[:, :size]**2 + rows[:, size:]**2
    rms = np.sqrt(energy.reshape(2500, 3, -1).mean(axis=(0, 2)))
    scale = np.tile(np.repeat(rms, size // 3), 2)
    scaled = rows / scale
    eigenvalues, vectors = np.linalg.eigh(scaled.T @ scaled)
    keep = eigenvalues > 1e-12 * eigenvalues[-1]
    transform = vectors[:, keep] / np.sqrt(eigenvalues[keep])
    atoms = np.ascontiguousarray(scaled @ transform)
    return atoms, (atoms**2).sum(axis=1), scale, transform


def sparse_layout(target, count):
    """Match a small number of active or inactive cells to the observed complex field."""
    number = min(count, 2500 - count)
    if not 1 <= number <= 32:
        return None
    _, indices, grid = receiver_rows()
    fields = target[indices]
    plane = fields[:, grid[:, None], grid].reshape(len(indices), -1)
    region = roi(fields).reshape(len(indices), -1)
    observed = np.concatenate((plane, region), axis=1).ravel()
    atoms, norm2, scale, transform = sparse_dictionary()
    residual = (np.r_[observed.real, observed.imag] / scale) @ transform
    if number != count:
        residual = atoms.sum(axis=0) - residual
    chosen = np.zeros(2500, dtype=bool)
    locations = []
    def place():
        gain = 2 * (atoms @ residual) - norm2
        gain[chosen] = -np.inf
        cell = int(np.argmax(gain))
        chosen[cell] = True
        residual[:] -= atoms[cell]
        return cell
    for _ in range(number):
        locations.append(place())
    for _ in range(14):
        previous = chosen.copy()
        for slot, cell in enumerate(locations):
            residual += atoms[cell]
            chosen[cell] = False
            locations[slot] = place()
        if np.array_equal(previous, chosen):
            break
    return (chosen if number == count else ~chosen).reshape(50, 50).astype(float)


def standardize(a):
    return (a - a.mean()) / max(a.std(), 1e-12)


def select_count(scores, count):
    flat = scores.ravel()
    chosen = np.argpartition(flat, flat.size - count)[-count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(scores.shape)


def cluster_centres(density, count, rng):
    # Match the density to a disk; normalizing the clipped disk area also tests corners.
    radius = max(2., np.sqrt(count / np.pi))
    axis = np.minimum(np.arange(100), 100 - np.arange(100))
    disk = (axis[:, None]**2 + axis**2 <= radius**2).astype(float)
    kernel = np.fft.rfft2(disk)
    def totals(values):
        return np.fft.irfft2(np.fft.rfft2(values, s=(100, 100), axes=(0, 1)) * kernel,
                             s=(100, 100), axes=(0, 1))[:50, :50]
    density_mean = totals(density) / np.maximum(totals(np.ones_like(density)), 1.)
    for _ in range(6):
        cy, cx = np.unravel_index(np.argmax(density_mean), density_mean.shape)
        yield cy, cx
        density_mean[(Y - cy)**2 + (X - cx)**2 < max(9., radius**2 / 4)] = -np.inf
    yield from rng.uniform(1, 48, size=(18, 2))


def stripe_waves(density, rng):
    # The argument of the matched Fourier component gives stripe phase in array coordinates.
    size = 128
    components = np.fft.rfft2(density, s=(size, size), axes=(0, 1))
    ky = 2 * np.pi * np.fft.fftfreq(size)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(size)[None, :]
    radius = np.hypot(ky, kx)
    strength = np.where((radius >= .28) & (radius <= 1.14), abs(components), -np.inf)
    for _ in range(10):
        iy, ix = np.unravel_index(np.argmax(strength), strength.shape)
        wy, wx = ky[iy, 0], kx[0, ix]
        yield np.cos(wy * Y + wx * X + np.angle(components[iy, ix]))
        strength[(ky - wy)**2 + (kx - wx)**2 < .11**2] = -np.inf
    for _ in range(18):
        theta, omega, phase = rng.uniform(0, np.pi), rng.uniform(.3, 1.1), rng.uniform(0, 2 * np.pi)
        yield np.cos(omega * (np.sin(theta) * Y + np.cos(theta) * X) + phase)


def shape_candidates(family, density, best):
    k = int(best.sum())
    rng = np.random.default_rng(310400 + FAMILIES.index(family))
    fit = np.clip(standardize(density), -5., 5.)
    for view in (best, best[:, ::-1]):
        for turns in range(4):
            yield np.rot90(view, turns)
    if family == "cluster":
        for cy, cx in cluster_centres(fit, k, rng):
            shape = -np.hypot(Y - cy, X - cx)
            noise = rng.normal(size=fit.shape)
            for amplitude in (.8, 2.7, 5.):
                yield select_count(shape + amplitude * fit + 1.2 * noise, k)
    elif family == "stripe":
        for wave in stripe_waves(fit, rng):
            noise = .17 * rng.normal(size=fit.shape)
            for amplitude in (.08, .25, .55):
                yield select_count(wave + noise + amplitude * fit, k)
    elif family == "edge":
        for _ in range(24):
            noise = rng.normal(size=fit.shape) * rng.uniform(.8, 3.)
            for amplitude in (.8, 2.7, 5.):
                yield select_count(-EDGE + noise + amplitude * fit, k)
    elif family == "patch":
        for length in (1.4, 2., 2.8, 4.):
            target_smooth = standardize(smooth_noise(fit, length))
            for _ in range(6):
                lumps = standardize(smooth_noise(rng.normal(size=fit.shape), length))
                for amplitude in (.4, 1.4, 4.):
                    yield select_count(lumps + amplitude * target_smooth + .15 * fit, k)


def sampled_levels(planes, regions, goal_plane, goal_region):
    e = abs(planes) - abs(goal_plane)
    r = abs(regions) - abs(goal_region)
    return np.stack([np.sqrt((e**2).mean(axis=(-2, -1))), abs(e[:, 0].mean(axis=-1)),
                     abs(r[:, 0, 12]), abs(r[:, 2, 12]),
                     np.sqrt((r[:, 0]**2).mean(axis=-1)), np.sqrt((r[:, 2]**2).mean(axis=-1)),
                     np.sqrt((e[:, 2]**2).mean(axis=-1))], axis=1)


def geometry_candidates(family, density, best_mask, source, objective, initial_p):
    """Screen distinct geometries; the caller certifies class and solves all 23 frequencies."""
    if family == "random":
        return
    rows, indices, grid = receiver_rows()
    masks = np.asarray(list(shape_candidates(family, density, best_mask)), dtype=bool)
    # Keep poses distinct, retaining generator order for deterministic ties.
    _, first = np.unique(np.packbits(masks.reshape(len(masks), -1), axis=1), axis=0, return_index=True)
    masks = masks[np.sort(first)]
    planes = masks.reshape(len(masks), -1).astype(float) @ rows
    real, imag = np.split(planes, 2, axis=1)
    receivers = (real + 1j * imag).reshape(len(masks), len(indices), -1)
    dim = len(grid)**2
    def sample(p):
        fields = p[indices]
        return fields[:, grid[:, None], grid].reshape(len(indices), -1), roi(fields).reshape(len(indices), -1)
    target, region = sample(objective.target)
    start, start_region = sample(initial_p)
    bounds = sampled_levels(start[None], start_region[None], target, region)[0]
    values = sampled_levels(receivers[:, :, :dim], receivers[:, :, dim:], target, region)
    ratio = values / np.maximum(bounds, 1e-10)
    over = np.maximum(ratio - 1., 0.)
    order = np.lexsort((ratio[:, -1], np.log1p(over).sum(axis=1), (ratio > 1).sum(axis=1)))
    for index in order:
        if not np.array_equal(masks[index], best_mask):
            yield masks[index].astype(float)
