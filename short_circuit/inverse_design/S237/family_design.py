"""Class-constrained inverse design on the native open-circuit PMUT equations."""
from functools import lru_cache
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent.parent
for folder in ("S217", "S232", "S284"):
    sys.path.insert(0, str(BASE / folder))

from model import FREQ, FB, G, N, conv, forward, mask_array, np, parse_goals, solve
from common import H, P, roi, score
from bank import FAMILIES, make_mask
from stable_classifier import StableClassifier
from point_swaps import point_proposals


@lru_cache(maxsize=1)
def classifier():
    return StableClassifier()


def certified(mask, family):
    checker = classifier()
    k = int(np.sum(mask))
    # Bound the lifetime cost of requests for many distinct, non-tabulated K.
    if k not in checker.extra and len(checker.extra) >= 12:
        oldest = next(iter(checker.extra))
        checker.extra.pop(oldest)
        checker.extra_orbits.pop(oldest)
    result = checker.predict(mask)
    return result["status"] == "single" and result["candidates"] == [family]


@lru_cache(maxsize=12)
def common_forward(packed, frequencies):
    mask = np.unpackbits(np.frombuffer(packed, np.uint8), count=2500).reshape(50, 50).astype(float)
    return forward(mask, frequencies)


def starting_field(mask, frequencies):
    return common_forward(np.packbits(mask.astype(np.uint8)).tobytes(), tuple(frequencies))


class PointObjective:
    def __init__(self, goals, weighted):
        self.ks, self.goals = parse_goals(goals)
        self.used = np.isfinite(self.goals)
        energy = np.nansum(self.goals**2, axis=(1, 2))
        if not np.isfinite(energy).all():
            raise ValueError("目标幅值过大")
        if weighted and np.any(energy <= 0):
            raise ValueError("逐频归一化要求各目标频率至少有一个非零声压。")
        self.weight = (1 / energy if weighted else np.ones(len(energy)))[:, None, None]
        self.target = np.nan_to_num(self.goals)

    def estimate_count(self, full):
        visible = np.where(self.used, np.abs(full), 0.)
        return 2500 * np.sum(self.weight * visible * self.target) / max(np.sum(self.weight * visible**2), 1e-20)

    def evaluate(self, p):
        residual = np.where(self.used, abs(p) - self.target, 0.)
        values = abs(residual[self.used])
        norm = np.linalg.norm(self.target[self.used])
        maximum = float(values.max())
        metrics = dict(items=int(values.size), passed=int((values <= 10).sum()),
                       rmse=float(np.sqrt(np.mean(values**2))), maximum=maximum,
                       relative=float(100*np.linalg.norm(values)/norm) if norm else None,
                       objective=float(np.sum(self.weight * residual**2)))
        return (maximum > 10, metrics["objective"]), metrics

    def derivatives(self, p):
        return np.where(self.used, self.weight * (abs(p) - self.target), 0.)

    def finish(self, metrics, initial):
        metrics["initial_rmse"] = initial["rmse"]
        ok = metrics["passed"] == metrics["items"]
        reason = f"原方程复算有 {metrics['passed']}/{metrics['items']} 项幅值误差不超过 10 Pa。"
        return ok, reason


class ProtectedObjective:
    def __init__(self, target):
        self.ks = np.arange(len(FREQ))
        self.target = np.asarray(target)
        if self.target.shape != (len(FREQ), 50, 50) or not np.isfinite(self.target).all():
            raise ValueError("保护目标应为 23×50×50 的有限复声压。")
        self.a = abs(self.target)
        self.b = abs(roi(self.target))
        self.norm = np.linalg.norm(self.target)
        self.high_norm = np.linalg.norm(self.a[H])
        if not np.isfinite([self.norm, self.high_norm]).all() or min(self.high_norm, self.norm) <= 0:
            raise ValueError("保护目标的 12.50 MHz 整面幅值须有限且非零。")

    def baseline(self, p):
        self.bounds, self.before, self.peak = score(p, self.a, self.b)
        peak_index = int(np.flatnonzero(FREQ == self.peak)[0])
        self.probe = np.array(list(dict.fromkeys([P, H, peak_index])))
        self.rest = np.setdiff1d(self.ks, self.probe)

    def cannot_beat(self, probe_field, rank):
        """Reject only when the evaluated frequencies already preclude a better rank."""
        # Lower bounds are rounded down before testing strict inequalities.
        # In particular, the full-frequency amplitude error includes all the
        # squared errors from the probe. Its other summands are nonnegative.
        amplitude = abs(probe_field)
        errors = amplitude - self.a[self.probe]
        high = 100 * np.linalg.norm(errors[1]) / self.high_norm
        if not rank[0] and high > rank[3] + 1e-10 * (1 + rank[3]):
            return True
        local = abs(roi(probe_field[:2])) - self.b[[P,H]]
        limits = np.array([100 * np.linalg.norm(errors) / self.norm, abs(errors[0].mean()),
                           *abs(local[:,2,2]), *np.sqrt((local**2).mean(axis=(1,2)))])
        lower = np.maximum(0., limits - 1e-10 * (1 + limits))
        # The original discrete peak must exceed or tie every probe frequency.
        levels = amplitude.mean(axis=(1,2))
        peak_level = levels[np.flatnonzero(FREQ[self.probe] == self.peak)[0]]
        lost_peak = levels.max() > peak_level + 1e-10 * (1 + peak_level)
        violations = np.count_nonzero(lower > self.bounds) + int(lost_peak)
        if violations == 0:
            return False
        if not rank[0] or violations > rank[1]:
            return True
        if violations == rank[1]:
            over = np.maximum(lower - self.bounds, 0) / np.maximum(self.bounds, 1e-9)
            return over.sum() > rank[2] + 1e-10 * (1 + rank[2])
        return False

    def contender_field(self, candidate, best_rank, *, reusable_seed=False, pair_limits=None):
        from protected_linear import forward as search_forward, seed_field
        predict = seed_field if reusable_seed else search_forward
        if best_rank is None and pair_limits is None:
            return predict(candidate, self.ks)
        p, u = predict(candidate, self.probe)
        if pair_limits is not None:
            # Every full-frequency squared error includes these probe errors.
            # An exact 5.6 mean and a lower bound on the whole error can thus
            # rule out both complete acceptance and a two-metric Pareto improvement.
            e = abs(p) - self.a[self.probe]
            lower = np.array([100*np.linalg.norm(e)/self.norm, abs(e[0].mean())])
            lower -= 1e-7*(1+lower)
            if np.any(lower > pair_limits) and np.any(lower > self.bounds[:2]):
                return None
        elif self.cannot_beat(p, best_rank):
            return None
        field = np.empty_like(self.target, dtype=complex)
        flow = np.empty_like(field)
        field[self.probe], flow[self.probe] = p, u
        field[self.rest], flow[self.rest] = predict(candidate, self.rest)
        return field, flow

    def estimate_count(self, full):
        visible = abs(full[H])
        return 2500 * np.sum(visible*self.a[H]) / max(np.sum(visible**2), 1e-20)

    def evaluate(self, p):
        limits, high, peak = score(p, self.a, self.b)
        kept = limits <= self.bounds
        peak_ok = bool(peak == self.peak)
        complex_relative = float(100*np.linalg.norm(p-self.target)/self.norm)
        exact = complex_relative <= 1e-9
        passed = peak_ok and bool(kept.all()) and (high < self.before or exact)
        # Among partial results prefer satisfying the original constraints, then
        # smaller relative overshoots. This ordering never changes acceptance.
        violations = int((~kept).sum()) + int(not peak_ok)
        over = float(np.maximum((limits-self.bounds)/np.maximum(self.bounds, 1e-9), 0.).sum())
        rank = (not passed, violations if not passed else 0, over if not passed else 0., float(high))
        metrics = dict(before=float(self.before), after=float(high), initial_peak_MHz=float(self.peak),
                       peak_MHz=float(peak), peak_ok=peak_ok, limits_before=self.bounds.tolist(),
                       limits_after=limits.tolist(), limits_ok=kept.tolist(), exact_field=exact,
                       protected_ok=bool(passed), whole_relative_pct=float(limits[0]),
                       complex_relative_pct=complex_relative)
        return rank, metrics

    def derivatives(self, p):
        diff = np.zeros_like(p.real)
        diff[H] = abs(p[H]) - self.a[H]
        return diff

    def finish(self, metrics, initial):
        if metrics["protected_ok"] and metrics["exact_field"]:
            return True, "已数值复现完整目标（复声压整面全频相对误差≤10⁻⁹%），无需继续降低。"
        ok = metrics["protected_ok"]
        return ok, ("相对原始开关图，峰位保持、六项上限全部满足，12.50 MHz 整面相对误差下降。"
                    if ok else "本次同类搜索未同时满足原始开关图的峰位、六项上限与高频改善要求。")


def search_counts(original_k, estimate, fixed, levels):
    if fixed:
        return [original_k] if levels.min() <= original_k <= levels.max() else []
    estimate = int(np.clip(round(estimate), levels.min(), levels.max()))
    close = sorted(map(int, levels), key=lambda k: abs(k-estimate))
    options = [estimate, original_k, *close[:2]]
    return list(dict.fromkeys(k for k in options if levels.min() <= k <= levels.max()))


def proposals(mask, gradient):
    on = np.flatnonzero(mask)
    off = np.flatnonzero(1-mask)
    down = on[np.argsort(-gradient.ravel()[on], kind="stable")]
    up = off[np.argsort(gradient.ravel()[off], kind="stable")]
    for size, offset in ((24, 0), (6, 0), (2, 0), (1, 0), (1, 1), (1, 2), (2, 2)):
        if min(len(down), len(up)) < size + offset:
            continue
        trial = mask.copy()
        trial.flat[down[offset:offset+size]] = 0
        trial.flat[up[offset:offset+size]] = 1
        yield trial


def scalar_options(body):
    keys = ("scope", "point", "metric", "direction", "value", "tolerance")
    return {key: body[key] for key in keys if key in body}


def design_family(mask, family, mode, *, goals=None, fixed=False, target=None, parallel=False,
                  scope=None, point=None, metric="peak_Pa", direction="maximize", value=None, tolerance=None):
    if family not in FAMILIES or mode not in ("absolute", "weighted", "protected", "peak_band", "scalar") or type(fixed) is not bool:
        raise ValueError("无效的阵列类别、目标模式或通电数约束。")
    if mode == "scalar":
        from scalar_design import design_scalar
        return design_scalar(mask, family, scope="center" if scope is None else scope, point=point,
                             metric=metric, direction=direction, value=value, tolerance=tolerance)
    if mode == "peak_band":
        from peak_band import design_peak_band
        return design_peak_band(mask, family, target, scope="plane" if scope is None else scope, point=point)
    source = mask_array(mask)
    initial_k = int(source.sum())
    objective = ProtectedObjective(target) if mode == "protected" else PointObjective(goals, mode == "weighted")
    started = time.perf_counter()
    wall_deadline = started + (60 if parallel else 12)
    cpu_deadline = time.process_time() + 12 if parallel else None

    def expired():
        return time.perf_counter() >= wall_deadline or (parallel and time.process_time() >= cpu_deadline)
    if fixed and not 6 <= initial_k <= 2494:
        return dict(family=family, status="unavailable", mask=None, metrics={"active": initial_k},
                    reason=f"固定 {initial_k} 路不在五类几何判据的有效范围 6–2494 路内。")
    initial_p, initial_u = starting_field(source, objective.ks)
    if mode == "protected":
        objective.baseline(initial_p)
    far_cluster = False
    exclusion = None
    if mode == "protected" and fixed and family == "cluster":
        from cluster_bound import mean_exclusion
        from cluster_search import remote_mean
        exclusion = mean_exclusion(initial_k, float(objective.a[P].mean()), float(objective.bounds[1]))
        far_cluster = exclusion is not None or remote_mean(objective, initial_k)
    initial_rank, initial_metrics = objective.evaluate(initial_p)
    best_rank, best_mask, best_p, best_u, best_metrics = None, None, None, None, None
    attempts = improvements = 0
    best_reference = True
    if certified(source, family):
        best_rank, best_mask, best_p, best_u, best_metrics = initial_rank, source, initial_p, initial_u, initial_metrics

    def done():
        if best_metrics is None:
            return False
        return best_metrics["protected_ok"] if mode == "protected" else best_metrics["passed"] == best_metrics["items"]

    def consider(candidate, *, preserve_plane_errors=False, reusable_seed=False, require_pass=False,
                 cluster_pair=False, guarded_repair=False, high_only=False, low_mean=False):
        nonlocal attempts, best_rank, best_mask, best_p, best_u, best_metrics, best_reference
        if not certified(candidate, family):
            return False
        attempts += 1
        if guarded_repair or high_only or low_mean:
            from cluster_repair import protected_rank
            field, flow = starting_field(candidate, objective.ks)
            rank, metrics = objective.evaluate(field)
            if low_mean:
                from cluster_low_seeds import preserves
                if not preserves(metrics,best_metrics):
                    return False
            elif high_only:
                if not (all(metrics["limits_ok"]) and metrics["peak_ok"]
                        and metrics["after"] < best_metrics["after"]):
                    return False
            elif (not all(metrics["limits_ok"][:2]) or
                  (not metrics["protected_ok"] and protected_rank(metrics) >= protected_rank(best_metrics))):
                return False
            best_rank, best_mask, best_p, best_u, best_metrics = rank, candidate, field, flow, metrics
            best_reference = True
            return True
        if mode == "protected":
            # The cluster rescue tracks the two requested errors jointly. A
            # partial result can improve both despite a lower legacy rank.
            pair_limits = np.asarray(best_metrics["limits_after"][:2]) if cluster_pair else None
            waves = objective.contender_field(candidate, best_rank, reusable_seed=reusable_seed,
                                              pair_limits=pair_limits)
            if waves is None:
                return False
            field, flow = waves
        else:
            field, flow = forward(candidate, objective.ks)
        rank, metrics = objective.evaluate(field)
        reference = mode != "protected"
        if mode == "protected":
            # Check a proposed pass and numerical ties with the original solver.
            # This keeps strict improvement and exact reproduction anchored to
            # the identical reference operator and iteration used for the start.
            limits = np.asarray(metrics["limits_after"])
            close = np.all(limits <= objective.bounds + 1e-7 * (1 + objective.bounds))
            if close and (metrics["after"] <= objective.before + 1e-7 * (1 + objective.before)
                          or metrics["complex_relative_pct"] < 1e-7):
                field, flow = starting_field(candidate, objective.ks)
                rank, metrics = objective.evaluate(field)
                reference = True
        if require_pass and not metrics["protected_ok"]:
            return False
        if preserve_plane_errors and best_metrics is not None:
            newly_verified = metrics["protected_ok"] and not best_metrics["protected_ok"]
            if not newly_verified and (metrics["after"] > best_metrics["after"] or
                                       metrics["whole_relative_pct"] > best_metrics["whole_relative_pct"]):
                return False
        better = best_rank is None or rank < best_rank
        if cluster_pair and not metrics["protected_ok"]:
            prior = np.asarray(best_metrics["limits_after"][:2])
            current = np.asarray(metrics["limits_after"][:2])
            better = (not best_metrics["protected_ok"] and bool(np.all(current <= prior))
                      and bool(np.any(current < prior)))
        if better:
            best_rank, best_mask, best_p, best_u, best_metrics = rank, candidate, field, flow, metrics
            best_reference = reference
            return True
        return False

    def repair_protected(rounds):
        nonlocal improvements
        from protected_swaps import proposals as protected_proposals
        rng = np.random.default_rng(310000 + FAMILIES.index(family))
        for _ in range(rounds):
            if best_mask is None or done() or expired():
                break
            moved = False
            for i, candidate in enumerate(protected_proposals(best_mask, objective, best_p, best_u, family, rng)):
                if i >= 9 or expired():
                    break
                if consider(candidate):
                    improvements += 1
                    moved = True
                    break
            if not moved:
                break

    if mode == "protected" and fixed and not done():
        from geometry_screen import sparse_layout
        sparse = sparse_layout(objective.target, initial_k)
        if sparse is not None and consider(sparse, require_pass=True):
            improvements += 1

    early_pair = False
    sparse_seeds = None
    if (mode == "protected" and fixed and family == "cluster" and best_mask is None
            and exclusion is None and 6 <= initial_k <= 24 and not expired()):
        from sparse_cluster_seeds import select_seeds
        from cluster_transfer import field as proposal_pressure
        sparse_seeds = select_seeds(objective,initial_p,initial_k)
        # Identify a feasible starting point before spending the budget on
        # unconstrained updates. Both bounds are rechecked on the native grid.
        for m in sparse_seeds[0]:
            approx = objective.evaluate(proposal_pressure(m))[1]
            if not all(approx["limits_ok"][:2]) or not certified(m,family):
                continue
            p,u = starting_field(m,objective.ks)
            rank,verified = objective.evaluate(p)
            attempts += 1
            if all(verified["limits_ok"][:2]):
                best_rank,best_mask,best_p,best_u,best_metrics = rank,m,p,u,verified
                early_pair = True
                improvements += 1
                break

    # If a compact seed protects the two plane metrics but misses the high band,
    # approach the geometric boundary from the other side. Only an inferred
    # field already satisfying all acoustic gates is a valid starting point.
    if (early_pair and initial_k <= 12 and not done() and sparse is not None
            and best_metrics["after"] >= objective.before and not expired()
            and objective.evaluate(proposal_pressure(sparse))[1]["protected_ok"]):
        from cluster_continuation import proposals as continuation
        if parallel:
            # Worker CPU time includes the concurrent acoustic and geometry
            # screens. Keep the batch wall deadline while allowing both lanes.
            cpu_deadline += 16
        for candidate in continuation(objective,sparse,classifier(),expired):
            if consider(candidate,guarded_repair=True):
                improvements += 1
                if done():
                    break

    # A uniquely classified initial mask is already exactly at its own six
    # bounds. The ranked search can move off it only via a fully verified move.
    if mode == "protected" and best_mask is not None and not early_pair and not done():
        repair_protected(1)
    if not done() and not early_pair:
        estimate = initial_k
        if not fixed:
            full, _ = starting_field(np.ones_like(source), objective.ks)
            estimate = objective.estimate_count(full)
        counts = search_counts(initial_k, estimate, fixed, classifier().levels)
        rng = np.random.default_rng(287000 + FAMILIES.index(family))
        for k in counts:
            if expired() or done():
                break
            # Evaluate the first two uniquely classifiable seeds at this count.
            number_at_entry = attempts
            for _ in range(22):
                if expired():
                    break
                candidate, _ = make_mask(rng, k, family)
                consider(candidate.astype(float), reusable_seed=True)
                if done() or attempts - number_at_entry >= 2:
                    break
    for _ in range(0 if far_cluster or early_pair else 3):
        if best_mask is None or done() or expired():
            break
        grad = np.zeros_like(source)
        for j, error in enumerate(objective.derivatives(best_p)):
            if expired():
                break
            if not error.any():
                continue
            k = objective.ks[j]
            phase = np.divide(best_p[j], abs(best_p[j]), out=np.ones_like(best_p[j]), where=abs(best_p[j]) > 0)
            if mode == "protected":
                from protected_linear import solve as search_solve
            else:
                search_solve = solve
            adjoint = search_solve(best_mask, k, conv(G[k].conj(), error*phase), True)
            grad += np.real(adjoint.conj() * (N[k] + FB[k] * best_u[j]))
        improved = False
        for candidate in proposals(best_mask, grad):
            if expired():
                break
            if consider(candidate):
                improvements += 1
                improved = True
                break
        if not improved:
            break

    if mode != "protected" and best_mask is not None and objective.used.sum() <= 8:
        rng = np.random.default_rng(300001 + FAMILIES.index(family))
        for _ in range(8):
            if done() or expired():
                break
            moved = False
            for i, candidate in enumerate(point_proposals(best_mask, objective, best_p, best_u, family, rng)):
                if i >= 15 or expired():
                    break
                if consider(candidate):
                    improvements += 1
                    moved = True
                    break
            if not moved:
                break

    if mode == "protected" and best_mask is not None and not early_pair and not best_metrics["exact_field"] and not expired():
        from field_seed import drive_score, ranked_masks

        target_density = drive_score(objective.target)
        for step in range(1 if far_cluster else 3):
            if expired() or best_metrics["exact_field"] or (step > 0 and done()):
                break
            density = target_density if step == 0 else drive_score(objective.target - best_p)
            moved = False
            for candidate in ranked_masks(density, best_mask):
                if expired():
                    break
                if consider(candidate, preserve_plane_errors=True):
                    improvements += 1
                    moved = True
                if done():
                    break
            if step > 0 and not moved:
                break
        if not done() and not expired() and family != "random" and not far_cluster:
            from geometry_screen import geometry_candidates
            trial_start = attempts
            for i, candidate in enumerate(geometry_candidates(family, target_density, best_mask,
                                                             source, objective, initial_p)):
                if i >= 20 or attempts - trial_start >= 3 or expired():
                    break
                if consider(candidate):
                    improvements += 1
                    if done():
                        break

    if mode == "protected" and not done() and best_mask is not None and not far_cluster and not early_pair:
        repair_protected(2)

    if mode == "protected" and family == "cluster" and best_mask is not None and not early_pair and not done() and not expired():
        from cluster_search import candidates
        for candidate in candidates(objective, initial_p, best_mask, certified, expired):
            if expired():
                break
            if consider(candidate, cluster_pair=True):
                improvements += 1
                if done():
                    break

    if (mode == "protected" and family == "cluster" and fixed and 6 <= initial_k <= 24
            and best_mask is not None and not done() and exclusion is None and not expired()):
        from sparse_cluster_seeds import select_seeds
        from cluster_sparse import sparse_refinements
        seeds,near = sparse_seeds if sparse_seeds is not None else select_seeds(objective,initial_p,initial_k)
        for seed in ([] if early_pair else seeds):
            if expired() or done():
                break
            if consider(seed,cluster_pair=True):
                improvements += 1
        if near and (not early_pair or best_metrics["after"] < objective.before) and not done() and not expired():
            for candidate in sparse_refinements(objective,initial_p,seeds,certified,expired):
                if expired():
                    break
                if consider(candidate,cluster_pair=True):
                    improvements += 1
                    if done():
                        break

    if (mode == "protected" and fixed and family == "cluster" and 6 <= initial_k <= 24
            and exclusion is None and best_mask is not None and not done()
            and all(best_metrics["limits_ok"][:2])):
        from cluster_repair import proposals as cluster_repairs
        # The repair shares the existing request budget. Each accepted move
        # retains both limiting plane metrics in the native solve.
        for candidate in cluster_repairs(objective,best_mask,best_metrics,sparse,classifier(),expired):
            if consider(candidate, guarded_repair=True):
                improvements += 1
                if done():
                    break

    if (mode == "protected" and fixed and family == "cluster" and 6 <= initial_k <= 24
            and best_mask is not None and not done() and best_metrics["peak_ok"]
            and all(best_metrics["limits_ok"]) and not expired()):
        from cluster_micro import proposals as high_repairs
        for candidate in high_repairs(objective,best_mask,best_p,best_metrics,expired):
            if consider(candidate,high_only=True):
                improvements += 1
                break
        if not done() and not expired() and initial_k <= 12:
            from cluster_subset import proposals as subset_repairs
            for candidate in subset_repairs(objective,best_mask,best_p,best_metrics,classifier(),expired):
                if consider(candidate,high_only=True):
                    improvements += 1

    if (mode == "protected" and fixed and family == "cluster" and initial_k == 12
            and exclusion is None and best_mask is not None and not done()
            and not best_metrics["limits_ok"][1] and not expired()):
        from cluster_low_seeds import proposals as low_mean_seeds
        for candidate in low_mean_seeds(objective,best_mask,best_metrics,expired):
            if consider(candidate,low_mean=True):
                improvements += 1
                break

    if best_mask is None:
        return dict(family=family, status="unavailable", mask=None,
                    metrics=dict(active=initial_k, elapsed=time.perf_counter()-started),
                    reason="限定搜索中未找到以该类为唯一候选的开关图。")
    if fixed and int(best_mask.sum()) != initial_k:
        raise RuntimeError("类别搜索未保持固定通电数")
    if mode == "protected" and not best_reference:
        field, _ = starting_field(best_mask, objective.ks)
        _, best_metrics = objective.evaluate(field)
    ok, reason = objective.finish(best_metrics, initial_metrics)
    if exclusion is not None and not ok:
        best_metrics["cluster_mean_exclusion"] = exclusion
        reason = (f"在当前模型、固定 {initial_k} 路和唯一成团条件下，5.60 MHz 整面平均幅值"
                  f"至少为 {exclusion['lower_pa']:.3f} Pa；此目标允许的最高值为 "
                  f"{exclusion['upper_pa']:.3f} Pa。返回图已按全部 23 频复算。")
    best_metrics.update(active=int(best_mask.sum()), initial_active=initial_k, changed=int((source != best_mask).sum()),
                        trials=attempts, accepted=improvements, elapsed=time.perf_counter()-started)
    return dict(family=family, status="verified" if ok else "partial", mask=best_mask.astype(int).tolist(),
                metrics=best_metrics, reason=reason)
