"""Run the workbench's numerical operations in the browser worker."""
import base64
import json
from pathlib import Path
import sys

ROOT = Path("/pmut")
sys.path[:0] = [str(ROOT / name) for name in ("S237", "S217")]
from model import AXIS, FREQ, forward, mask_array, np


def encode(array):
    return base64.b64encode(array.astype("<c16", copy=False).tobytes()).decode("ascii")


def dispatch(path, payload):
    body = json.loads(payload)
    if path == "/config":
        mask = np.loadtxt(ROOT / "S217/start.csv", delimiter=",", dtype=int)
        result = dict(frequency=FREQ.tolist(), axis=AXIS.round(5).tolist(), mask=mask.tolist())
    elif path == "/example":
        result = dict(target=encode(np.load(ROOT / "S233/S005525_ref.npy")),
                      mask=np.loadtxt(ROOT / "S233/S005525_20261123_base.csv",
                                      delimiter=",", dtype=int).tolist(),
                      candidate=np.loadtxt(ROOT / "S234/S005525_20261123_visit38.csv",
                                           delimiter=",", dtype=int).tolist())
    elif path == "/predict":
        mask = mask_array(body["mask"])
        pressure, velocity = forward(mask, range(len(FREQ)))
        result = dict(mask=mask.astype(int).tolist(), metrics=None,
                      field=encode(pressure), velocity=encode(velocity))
    elif path == "/family-design":
        from family_design import design_family, scalar_options
        mode = body["mode"]
        target = None
        if mode in ("protected","peak_band"):
            target = np.frombuffer(base64.b64decode(body["target"], validate=True),
                                   dtype="<c16").reshape(len(FREQ), 50, 50)
        options = scalar_options(body) if mode == "scalar" else dict(scope=body.get("scope","plane"), point=body.get("point"))
        result = design_family(mask_array(body["mask"]), body["family"], mode,
                               fixed=True if mode in ("protected","peak_band","scalar") else body["fixed"],
                               goals=body.get("goals"), target=target, **options)
    else:
        raise ValueError("未知工作台操作")
    return json.dumps(result, ensure_ascii=False, allow_nan=False)
