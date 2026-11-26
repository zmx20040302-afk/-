"""Fast linear pressure estimates for ranking candidate compact arrays."""
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def table():
    with np.load(Path(__file__).with_suffix(".npz")) as data:
        return (data["basis"],data["canonical"],data["poses"],
                [data[f"c{k}"] for k in range(23)])


def _rotate(coefficients, pose, signs):
    if pose >= 4:
        coefficients = coefficients * signs
    r = pose % 4
    if r == 1:
        return coefficients.swapaxes(-2,-1) * signs[:,None]
    if r == 2:
        return coefficients * signs * signs[:,None]
    if r == 3:
        return coefficients.swapaxes(-2,-1) * signs
    return coefficients


@lru_cache(maxsize=1)
def source_bands():
    """Contiguous source coefficients for batched sparse pressure rankings."""
    basis, canonical, poses, bands = table()
    positions = [np.flatnonzero(poses == i) for i in range(8)]
    result = []
    for band in bands:
        d = band.shape[-1]
        signs = (1 - 2*(np.arange(d)%2)).astype(np.float32)
        coefficients = np.empty((2500,d,d),np.complex64)
        for i,indices in enumerate(positions):
            coefficients[indices] = _rotate(band[canonical[indices]],i,signs)
        result.append((np.ascontiguousarray(basis[:,:d],dtype=np.complex64), coefficients))
    return result


def field(mask):
    basis, canonical, poses, bands = table()
    active = np.flatnonzero(mask)
    groups = [(p,canonical[active[poses[active] == p]]) for p in np.unique(poses[active])]
    out = np.empty((len(bands),50,50),np.complex128)
    for k,band in enumerate(bands):
        d = band.shape[-1]
        signs = 1. - 2.*(np.arange(d)%2)
        coeff = np.zeros((d,d),np.complex128)
        for pose,indices in groups:
            coeff += _rotate(band[indices].sum(axis=0,dtype=np.complex128),pose,signs)
        q = basis[:,:d]
        out[k] = q @ coeff @ q.T
    return out
