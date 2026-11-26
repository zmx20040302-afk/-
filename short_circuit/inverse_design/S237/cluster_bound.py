"""Conservative 5.60 MHz mean bounds for the verified classifier and physical operator."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import math


# S314/cluster_extrema/check_certificate.py independently replays all 99 cells
# without the optimizer. Each value is rounded down to a whole pascal.
_LOWER_PA = {12: 133, 150: 3164, 1250: 22094}
_FROZEN = {
    "S217/model.py": "026655dd28dfd805943ee9e04597c294e3fe9df7aabfeeda69ccc52dcc5bda70",
    "S217/kernels.npz": "97766c0b4fba1cdff6938356bb3e3d1d3b415c49355a35831e1ab2857eb27a41",
    "S232/common.py": "ddff18ebbc12f220e2b4983252902b37d19b0839457a368967f27f09583210e4",
    "S281/geometry.py": "ff5fe80d2ebf217da94eefb893504282c802c28455ea66119c8a4877af470a34",
    "S284/model.npz": "64eb183bb88322892bf6e30634ede160ceae8f0973950c947243ef9e90d00f97",
    "S284/classifier.py": "02f7a59058f31c2766dcd65b032e4715be81c0b7e33a3dc7aef7ac472ba6a2bf",
    "S284/stable_classifier.py": "d5211b70ce86993b97d17be09e68ec35365453f0fb1d7511376a913fa9765db0",
}


@lru_cache(maxsize=1)
def _applicable():
    base = Path(__file__).resolve().parents[1]
    try:
        return all(sha256((base / name).read_bytes()).hexdigest() == digest
                   for name, digest in _FROZEN.items())
    except OSError:
        return False


def mean_exclusion(k, target_mean_pa, allowed_bias_pa):
    lower = _LOWER_PA.get(k)
    upper = float(target_mean_pa + allowed_bias_pa)
    if lower is not None and math.isfinite(upper) and upper < lower and _applicable():
        return {"lower_pa": lower, "upper_pa": upper}
    return None
