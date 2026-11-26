"""Geometry classifier for a 50 by 50 PMUT switch mask. Only NumPy is needed."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "S281"))
import geometry as original

FAMILIES = original.FAMILIES
FEATURES = (*original.FEATURES, "minority_gyration_cells", "minority_axis_ratio",
            "densest_window9", "fine_stripe_peak", "high_frequency_peak",
            "band_directionality")
ZH = dict(zip(FAMILIES, ["随机分散", "紧凑成团", "边缘优先", "条纹", "不规则多斑块"]))
STATUS = {"single": "单一候选", "ambiguous": "多类相近", "outside": "超出参考范围",
          "conflict": "无一致候选", "insufficient": "少数态不足", "uniform": "均匀图"}
ALPHA = .10
fine_k = np.fft.fftfreq(150) * 50
ky, kx = np.meshgrid(fine_k, fine_k, indexing="ij")
radius = np.hypot(kx, ky)
band = (radius >= 2) & (radius <= 10)
high_band = radius > 12
direction = ((kx[band] + 1j * ky[band]) / radius[band]) ** 2


def validate(mask):
    mask = np.asarray(mask)
    if mask.shape != (50, 50) or not np.isin(mask, [0, 1]).all():
        raise ValueError("需要 50×50 无表头数值矩阵，每格为 0 或 1。")
    return mask.astype(bool)


def describe(mask):
    mask = validate(mask)
    base = original.features(mask)
    minority = mask if mask.sum() <= 1250 else ~mask
    n = int(minority.sum())
    y, x = np.nonzero(minority)
    centered = np.stack((x - x.mean(), y - y.mean()))
    inertia = np.linalg.eigvalsh(centered @ centered.T / n).clip(min=0)
    axis_ratio = inertia[0] / max(inertia[1], 1e-12)
    integral = np.pad(minority.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = integral[9:, 9:] + integral[:-9, :-9] - integral[:-9, 9:] - integral[9:, :-9]
    spectrum = np.abs(np.fft.fft2(minority - n / 2500, s=(150, 150), axes=(0, 1))) / n
    power_band = spectrum[band] ** 2
    anisotropy = abs(np.sum(power_band * direction)) / max(power_band.sum(), 1e-16)
    return np.r_[base, np.sqrt(inertia.sum()), axis_ratio, counts.max() / min(n, 81),
                 spectrum[band].max(), spectrum[high_band].max(), anisotropy]


def comparable(descriptors, k):
    # Remove density factors before transferring a rule to a nearby count.
    values = np.array(descriptors, copy=True)
    minority = min(k, 2500 - k)
    values[..., 0] *= k / minority
    values[..., 3] /= (k / 2500) * (1 - k / 2500)
    values[..., 4:6] *= (2500 - minority) / minority
    return values


def count_index(k, levels):
    log_odds = np.log(k / (2500 - k))
    return int(np.argmin(abs(np.log(levels / (2500 - levels)) - log_odds)))


def distances(value, reference, scales):
    delta = (reference - value) / scales
    d = np.linalg.norm(delta, axis=-1)
    return np.partition(d, 2, axis=1)[:, :3].mean(axis=1)


def relative(d):
    others = np.broadcast_to(d, (len(d), len(d))).copy()
    np.fill_diagonal(others, np.inf)
    return d / np.maximum(others.min(axis=1), 1e-12)


def prepare_stratum(ref, cal):
    scale = np.maximum(ref.std(axis=(0, 1)), 1e-8)
    scores = np.empty((len(FAMILIES), 2, cal.shape[1]))
    for i, examples in enumerate(cal):
        for j, x in enumerate(examples):
            d = distances(x, ref, scale)
            scores[i, :, j] = [d[i], relative(d)[i]]
    return ref, scale, scores


def generate_stratum(k):
    from bank import make_mask
    seen = set()
    groups = []
    for stream, count in [(0, 45), (1, 39)]:
        rng = np.random.default_rng(np.random.SeedSequence([284101, k, stream]))
        families = []
        for family in FAMILIES:
            values = []
            while len(values) < count:
                mask, _ = make_mask(rng, k, family)
                key = original.orbit(mask)
                if key not in seen:
                    seen.add(key)
                    values.append(comparable(describe(mask), k))
            families.append(values)
        groups.append(np.array(families))
    return prepare_stratum(*groups), seen


class Classifier:
    def __init__(self, model_path=HERE / "model.npz", nearest=False):
        self.nearest = nearest
        self.extra = {}
        self.extra_orbits = {}
        with np.load(model_path, allow_pickle=False) as model:
            self.levels = model["levels"]
            self.ref = model["references"]
            self.scales = model["scales"]
            self.cal = model["calibration"]

    def predict(self, mask):
        mask = validate(mask)
        k = int(mask.sum())
        output = dict(k=k, status="", family="", candidates=[], first="", second="",
                      calibration_k=None, exact_k=False, calibration_source="", features={}, distances={},
                      compatibility={})
        if k in (0, 2500):
            output["status"] = "uniform"
            return output
        raw = describe(mask)
        output["features"] = dict(zip(FEATURES, raw.tolist()))
        if k < self.levels.min() or k > self.levels.max():
            output["status"] = "insufficient"
            return output
        idx = count_index(k, self.levels)
        anchor = int(self.levels[idx])
        if k == anchor or self.nearest:
            ref, scale, calibration = self.ref[idx], self.scales[idx], self.cal[idx]
            source = "preset" if k == anchor else "nearest"
        else:
            if k not in self.extra:
                self.extra[k], self.extra_orbits[k] = generate_stratum(k)
            ref, scale, calibration = self.extra[k]
            anchor, source = k, "generated"
        d = distances(comparable(raw, k), ref, scale)
        scores = np.stack((d, relative(d)), axis=1)
        # Shape: class × score × calibration example. Large scores are unusual.
        p = (1 + (calibration >= scores[:, :, None]).sum(axis=2)) / (1 + calibration.shape[-1])
        compatibility = np.minimum(1, 2 * np.min(p, axis=1))
        ordered = np.argsort(d, kind="stable")
        keep = [FAMILIES[i] for i in ordered if compatibility[i] > ALPHA]
        if len(keep) == 1:
            status = "single"
        elif keep:
            status = "ambiguous"
        elif np.all(2 * p[:, 0] <= ALPHA):
            status = "outside"
        else:
            status = "conflict"
        output.update(status=status, candidates=keep, family=keep[0] if len(keep)==1 else "",
                      first=FAMILIES[ordered[0]], second=FAMILIES[ordered[1]],
                      calibration_k=anchor, exact_k=k==anchor, calibration_source=source,
                      distances=dict(zip(FAMILIES, d.tolist())),
                      compatibility=dict(zip(FAMILIES, compatibility.tolist())))
        return output


def main():
    parser = argparse.ArgumentParser(description="PMUT 五类开关图判别")
    parser.add_argument("mask", type=Path, help="无表头 50×50 二值 CSV")
    parser.add_argument("--json", action="store_true", help="输出完整判别和连续指标")
    parser.add_argument("--nearest", action="store_true", help="非预置通电数直接使用最近档校准")
    args = parser.parse_args()
    try:
        mask = np.loadtxt(args.mask, delimiter=",", encoding="utf-8-sig")
        result = Classifier(nearest=args.nearest).predict(mask)
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"通电 {result['k']} / 2500　{STATUS[result['status']]}")
        if result["first"]:
            scope = "同档校准" if result["exact_k"] else "相邻档迁移"
            print(f"参照 {result['calibration_k']} 路（{scope}）")
            print("可接受家族：" + ("、".join(ZH[f] for f in result["candidates"]) or "无"))
            print("距离最近两类：" + "、".join(ZH[result[label]] for label in ("first", "second")))


if __name__ == "__main__":
    main()
