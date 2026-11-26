"""Reproduce frozen decisions under square symmetries and include numerical score ties."""
import numpy as np

from classifier import Classifier, HERE, generate_stratum, original, validate


# A score originally equal to a stored calibration score can differ after the
# same real-valued calculation in another pixel traversal order. The allowance
# is applied only at the >= rank comparison, never to feature or distance values.
TIE_ATOL = 1e-12
TIE_RTOL = 1e-12


def inclusive_upper_bound(scores):
    return scores + (TIE_ATOL + TIE_RTOL * np.abs(scores))


class StableClassifier(Classifier):
    def __init__(self, model_path=HERE / "model.npz", nearest=False):
        super().__init__(model_path=model_path, nearest=nearest)
        self.cal = inclusive_upper_bound(self.cal)

    def predict(self, mask):
        data = validate(mask)
        packed = np.frombuffer(original.orbit(data), dtype=np.uint8)
        canonical = np.unpackbits(packed, count=data.size).reshape(data.shape)
        k = int(canonical.sum())
        if (not self.nearest and self.levels.min() <= k <= self.levels.max()
                and k not in self.levels and k not in self.extra):
            (ref, scale, scores), orbits = generate_stratum(k)
            self.extra[k] = ref, scale, inclusive_upper_bound(scores)
            self.extra_orbits[k] = orbits
        return super().predict(canonical)
