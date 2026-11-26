"""Search a certified fixed-count cluster bank and adapt its boundary to a target."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from common import P, roi
from geometry_screen import receiver_rows, sampled_levels
from model import FB, G, N, conv
from protected_linear import forward, solve


@lru_cache(maxsize=1)
def bank():
    with np.load(Path(__file__).with_name("cluster_pool.npz")) as data:
        return {key:data[key] for key in data.files}


def remote_mean(objective, count):
    """A search-budget heuristic based on observed witnesses, not a feasibility bound."""
    data = bank()
    means = data["means"][data["counts"] == count]
    if not means.size:
        return False
    goal = objective.a[P].mean()
    return goal + 2*objective.bounds[1] < means.min() or goal - 2*objective.bounds[1] > means.max()


def pose(a, i):
    return np.rot90(a if i < 4 else a[..., :, ::-1], i % 4, axes=(-2, -1))


def priorities(ratio):
    # The two named constraints determine the coarse frontier. Once both fit,
    # the remaining five errors and strict high-frequency improvement dominate.
    over = np.maximum(ratio - 1, 0)
    pair = np.log1p(over[:, :2])
    return np.where(pair.max(axis=1) > 0, 1 + pair.max(axis=1) + .3*pair.mean(axis=1),
                    .45*np.tanh(np.log1p(over[:, 2:]).sum(axis=1)) + .4*np.tanh(ratio[:, -1]))


class Screen:
    def __init__(self, objective, initial):
        self.rows, indices, grid = receiver_rows()
        self.dim = grid.size**2
        def samples(p):
            plane = p[indices][:, grid[:,None], grid]
            local = roi(p[indices])
            return plane.reshape(3,-1), local.reshape(3,-1)
        self.target_plane, self.target_roi = samples(objective.target)
        initial_plane, initial_roi = samples(initial)
        self.bounds = sampled_levels(initial_plane[None], initial_roi[None],
                                     self.target_plane, self.target_roi)[0]
        self.bounds[1] = objective.bounds[1]
        self.bounds = np.maximum(self.bounds, 1e-10)
        self.target_mean = objective.a[P].mean()

    def ratio(self, amplitudes, means):
        levels = sampled_levels(amplitudes[:, :, :self.dim], amplitudes[:, :, self.dim:],
                                self.target_plane, self.target_roi)
        levels[:, 1] = abs(means - self.target_mean)
        return levels / self.bounds

    def seeds(self, k):
        data = bank()
        subset = np.flatnonzero(data["counts"] == k)
        if not subset.size:
            return []
        amplitudes = data["sampled"][subset]
        plane = amplitudes[:, :, :self.dim].reshape(-1,3,8,8)
        local = amplitudes[:, :, self.dim:].reshape(-1,3,5,5)
        ratios = []
        for i in range(8):
            observations = np.concatenate([pose(plane,i).reshape(-1,3,64),
                                           pose(local,i).reshape(-1,3,25)],axis=2)
            ratios.append(self.ratio(observations,data["means"][subset]))
        ratios = np.concatenate(ratios)
        # Prioritize other protections as a tie-break across symmetric orientations.
        secondary = np.log1p(np.maximum(ratios-1.,0.)).sum(axis=1)
        order = np.lexsort((secondary, priorities(ratios)))
        chosen, seen = [], set()
        for index in order:
            orientation, pos = divmod(index, len(subset))
            if pos not in seen:
                original = np.unpackbits(data["packed"][subset[pos]],count=2500).reshape(50,50)
                chosen.append(pose(original,orientation).astype(float))
                seen.add(pos)
                if len(chosen) == 3:
                    break
        return chosen

    def walk(self, mask, certified, expired):
        # Use the exact 5.6 plane and its adjoint for the signed mean gradient.
        # The three-frequency receivers estimate the whole-plane error gradient.
        for step in range(5):
            if expired():
                break
            p,u = forward(mask,[P])
            magnitude = abs(p[0])
            mean = magnitude.mean()
            unit = p[0]/np.maximum(magnitude,1e-15)
            adj = solve(mask,P,conv(G[P].conj(),unit/mask.size),True)
            dmean = (adj.conj()*(N[P] + FB[P]*u[0])).real.ravel()
            flat = mask.ravel()
            real, imag = np.split(flat @ self.rows,2)
            pressure = (real + 1j*imag).reshape(3,-1)
            ratio = self.ratio(abs(pressure)[None],np.array([mean]))[0]
            residual = abs(pressure[:, :self.dim]) - abs(self.target_plane)
            receiver = np.zeros_like(pressure)
            receiver[:, :self.dim] = (residual * pressure[:, :self.dim] /
                np.maximum(abs(pressure[:, :self.dim]), 1e-15) /
                max(np.linalg.norm(residual)*np.sqrt(residual.size),1e-15))
            whole = self.rows @ np.r_[receiver.real.ravel(),receiver.imag.ravel()]
            dw = whole / self.bounds[0]
            dm = np.sign(mean-self.target_mean)*dmean/self.bounds[1]
            dw /= max(ratio[0],1.)
            dm /= max(ratio[1],1.)
            y, x = np.indices(mask.shape)
            cy, cx = y[mask.astype(bool)].mean(), x[mask.astype(bool)].mean()
            r = np.hypot(y-cy,x-cx).ravel()
            on, off = np.flatnonzero(flat), np.flatnonzero(1-flat)
            amounts = sorted({min(len(on), len(off), n) for n in (1,2,6,16,40,90)})
            trials = []
            for mixture in (.25,.65,1.):
                slope = dm*mixture + dw*(1.1-mixture)
                for penalty in (0., .06, .22, .65):
                    g = slope + penalty*np.std(slope)*r/max(2.,r[on].std())
                    down = on[np.argsort(-g[on])[:max(amounts)]]
                    up = off[np.argsort(g[off])[:max(amounts)]]
                    for size in amounts:
                        m = mask.copy()
                        m.flat[down[:size]], m.flat[up[:size]] = 0, 1
                        trials.append(m)
            trials = np.unique(np.asarray(trials).reshape(-1,2500),axis=0)
            delta = trials - flat
            a,b = np.split(delta @ self.rows,2,axis=1)
            amps = abs(pressure + (a+1j*b).reshape(-1,3,89))
            ratios = self.ratio(amps,mean + delta @ dmean)
            predicted = priorities(ratios)
            original = priorities(ratio[None])[0]
            candidate = None
            for i in np.argsort(predicted)[:20]:
                if predicted[i] >= original-1e-7 or expired():
                    break
                m = trials[i].reshape(50,50)
                if certified(m, "cluster"):
                    candidate = m
                    break
            if candidate is None:
                break
            mask = candidate
            yield mask


def candidates(objective, initial, incumbent, certified, expired):
    screen = Screen(objective, initial)
    seeds = screen.seeds(int(incumbent.sum()))
    # Off-grid counts use the existing certified incumbent as the initial shape.
    start = seeds[0] if seeds else incumbent
    if seeds:
        yield start
    intermediates = list(screen.walk(start, certified, expired))
    if intermediates:
        yield intermediates[-1]
    if len(seeds) > 1:
        yield seeds[1]
    if len(intermediates) > 2:
        yield intermediates[len(intermediates)//2]
