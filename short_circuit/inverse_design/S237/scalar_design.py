"""Search uniquely classified layouts for one spectral quantity, with selectable drive count."""
from itertools import islice
from math import isfinite
from time import perf_counter

from family_design import (FB, FREQ, FAMILIES, G, N, certified, classifier, conv, make_mask,
                           mask_array, np, proposals, solve, starting_field)
from peak_band import LEVEL, curve_receiver, features, spectrum


def _edge_indices(curve, peak, step):
    return next((j for j in range(peak+step, -1 if step < 0 else len(curve), step)
                 if curve[j] <= LEVEL*curve[peak]), None)


def _rank(curve, response, metric, target):
    actual = response[metric]
    if actual is not None:
        return (0, abs(actual-target) if target is not None else -actual)
    # A one-sided band cannot be compared to a measured width. Guide trials
    # towards two observed crossings using the remaining threshold excess.
    peak = int(curve.argmax())
    amplitude = curve[peak]
    gaps = [max(0., float(side.min()/amplitude)-LEVEL) if side.size and amplitude else 1.
            for side in (curve[:peak],curve[peak+1:])]
    return (sum(g > 0 for g in gaps) or 1, sum(gaps))


def _curve_derivative(curve, response, metric):
    """Derivative of the sampled peak or the two piecewise-linear crossings."""
    grad = np.zeros(len(curve))
    peak = int(curve.argmax())
    if metric == "peak_Pa":
        grad[peak] = 1.
        return grad
    if response["bandwidth_MHz"] is None:
        # Proposal surrogate for an open band: move each inaccessible flank
        # below the threshold or shift an endpoint peak into the sampled span.
        for step in (-1,1):
            if _edge_indices(curve,peak,step) is not None:
                continue
            side = np.arange(peak+step,-1 if step < 0 else len(curve),step)
            if side.size:
                grad[side[np.argmin(curve[side])]] -= 1.
                grad[peak] += LEVEL
            else:
                grad[peak] -= 1.
                grad[1+np.argmax(curve[1:-1])] += 1.
        return grad
    left, right = response["left_MHz"],response["right_MHz"]
    for step in (-1,1):
        far = _edge_indices(curve,peak,step)
        near = far-step
        dy = curve[near]-curve[far]
        t = (LEVEL*curve[peak]-curve[far])/dy
        scale = (FREQ[near]-FREQ[far])/dy
        # d(R-L) or d[200 (R-L)/(R+L)], in the selected units.
        multiplier = step
        if metric == "relative_bandwidth_pct":
            multiplier *= 400*(right if step < 0 else left)/(right+left)**2
        scale *= multiplier
        grad[peak] += scale*LEVEL
        grad[far] -= scale*(1-t)
        grad[near] -= scale*t
    return grad


def _mask_gradient(mask, field, flow, curve, response, metric, target, scope, point):
    weights = _curve_derivative(curve,response,metric)
    # proposals() minimizes. An open band always seeks measurable crossings.
    lower = target is not None and response[metric] is not None and response[metric] > target
    if not lower:
        weights *= -1
    grad = np.zeros(mask.shape)
    for k in np.flatnonzero(weights):
        receiver = curve_receiver(field[k],scope,point)
        adjoint = solve(mask,k,conv(G[k].conj(),receiver),True)
        grad += weights[k]*np.real(adjoint.conj()*(N[k]+FB[k]*flow[k]))
    return grad


def _number(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool) and isfinite(x) and x >= 0


def design_scalar(mask, family, *, scope="center", point=None, metric="peak_Pa",
                  direction="maximize", value=None, tolerance=None, count_mode="current", count=None):
    from cluster_transfer import field as estimate
    from geometry_screen import shape_candidates
    if metric not in ("peak_Pa","bandwidth_MHz","relative_bandwidth_pct") or family not in FAMILIES:
        raise ValueError("请选择有效的声学指标与排布类别。")
    if direction not in ("maximize","match"):
        raise ValueError("请选择尽可能提高或匹配给定值。")
    if direction == "match" and not (_number(value) and (tolerance is None or _number(tolerance))):
        raise ValueError("给定值须为有限非负数；允许偏差可留空或填有限非负数。")
    if count_mode not in ("current","auto","specified"):
        raise ValueError("请选择自动搜索通电数、指定通电数或沿用当前图。")
    if count_mode == "specified" and (type(count) is not int or not 6 <= count <= 2494):
        raise ValueError("指定通电数须为 6–2494 的整数。")
    target = float(value) if direction == "match" else None
    tol = float(tolerance) if direction == "match" and tolerance is not None else None
    start = perf_counter()
    source = np.zeros((50,50)) if mask is None and count_mode != "current" else mask_array(mask)
    initial_count = int(source.sum())
    required_count = count if count_mode == "specified" else initial_count if count_mode == "current" else None
    tested_counts, screened_counts = set(), set()
    meta = dict(metric=metric,direction=direction,target_value=target,tolerance=tol,
                count_mode=count_mode,requested_count=count if count_mode=="specified" else None,
                active=None,initial_active=initial_count,scope=scope,point=point if scope=="point" else None,
                frequency_MHz=FREQ.tolist())
    def unavailable(reason):
        meta.update(tested_counts=sorted(tested_counts),screened_counts=sorted(screened_counts))
        return dict(family=family,status="unavailable",mask=None,metrics=meta,reason=reason)
    if required_count is not None and not 6 <= required_count <= 2494:
        return unavailable(f"上图 {required_count} 路超出五类唯一分类的适用范围（6–2494 路）。")
    ks = tuple(range(len(FREQ)))
    baseline_field,baseline_flow = starting_field(source,ks)
    initial_curve = spectrum(baseline_field,scope,point)
    initial = features(initial_curve)
    initial_value = initial[metric]
    meta.update(initial_curve_Pa=initial_curve.tolist(), initial_response=initial, initial_value=initial_value)
    seen = set()
    rng = np.random.default_rng(341100+FAMILIES.index(family))
    best = None
    trials = 0
    deadline = perf_counter()+8.
    def record(m, waves):
        nonlocal best,trials
        tested_counts.add(int(m.sum()))
        curve = spectrum(waves[0],scope,point)
        response = features(curve)
        rank = _rank(curve,response,metric,target)
        trials += 1
        if best is None or rank < best[0]:
            best = (rank,m.copy(),waves,curve,response)

    def matched():
        return (tol is not None and best is not None and best[0][0]==0
                and best[0][1] <= tol)

    def screen(candidates, keep, diverse=False):
        shortlist = []
        for m in candidates:
            if perf_counter() >= deadline or (matched() and not diverse):
                break
            key = np.packbits(m.astype(bool)).tobytes()
            k = int(m.sum())
            if key in seen or (required_count is not None and k!=required_count):
                continue
            seen.add(key)
            if not certified(m,family):
                continue
            screened_counts.add(k)
            if best is None:
                record(m,starting_field(m,ks))
            else:
                curve = spectrum(estimate(m),scope,point)
                shortlist.append((_rank(curve,features(curve),metric,target),m))
        selected = 0
        for _,m in sorted(shortlist,key=lambda pair:pair[0]):
            if matched() and (not diverse or len(tested_counts)>=2):
                break
            if diverse and int(m.sum()) in tested_counts:
                continue
            record(m,starting_field(m,ks))
            selected += 1
            if selected >= keep or perf_counter() >= deadline:
                break

    if (6 <= initial_count <= 2494 and (required_count is None or initial_count==required_count)
            and certified(source,family)):
        seen.add(np.packbits(source.astype(bool)).tobytes())
        record(source,(baseline_field,baseline_flow))
    if required_count is None:
        # Cover the calibrated count strata first, then localize both K and
        # layout around the best native replay. All K share one time budget.
        levels = [int(k) for k in classifier().levels]
        def seeds():
            for _ in range(3):
                for k in levels:
                    yield make_mask(rng,k,family)[0]
        screen(seeds(),4,diverse=True)
    else:
        screen((make_mask(rng,required_count,family)[0] for _ in range(18)),2)
    for _ in range(4):
        if best is None or matched() or perf_counter() >= deadline:
            break
        prev,incumbent,waves,curve,response = best
        grad = _mask_gradient(incumbent,*waves,curve,response,metric,target,scope,point)
        count_now = int(incumbent.sum())
        if required_count is None:
            # Propose new counts between the calibrated levels. Screen these
            # acoustically before invoking potentially costly off-grid calibration.
            down = np.flatnonzero(incumbent)
            up = np.flatnonzero(incumbent==0)
            down = down[np.argsort(-grad.flat[down])]
            up = up[np.argsort(grad.flat[up])]
            lower = max((k for k in levels if k<count_now),default=6)
            upper = min((k for k in levels if k>count_now),default=2494)
            next_counts = {(count_now+lower)//2,(count_now+upper+1)//2,count_now-1,count_now+1}
            if metric=="peak_Pa" and target is not None and response[metric]>0:
                estimated = np.clip(count_now*(target/response[metric]),6,2494)
                next_counts.add(round(float(estimated)))
            exploratory = []
            for k in sorted(next_counts):
                if k==count_now or not 6<=k<=2494 or perf_counter()>=deadline:
                    continue
                m = incumbent.copy()
                if k>count_now:
                    m.flat[up[:k-count_now]] = 1
                else:
                    m.flat[down[:count_now-k]] = 0
                trial_curve = spectrum(estimate(m),scope,point)
                exploratory.append((_rank(trial_curve,features(trial_curve),metric,target),m))
            for _,m in sorted(exploratory,key=lambda row:row[0])[:2]:
                if perf_counter()>=deadline or matched():
                    break
                token = np.packbits(m.astype(bool)).tobytes()
                if token not in seen:
                    seen.add(token)
                    if certified(m,family):
                        screened_counts.add(int(m.sum()))
                        record(m,starting_field(m,ks))
        def variations():
            yield from proposals(incumbent,grad)
            # Extended swaps also move dense layouts an appreciable distance.
            down = np.flatnonzero(incumbent)
            up = np.flatnonzero(incumbent==0)
            down = down[np.argsort(-grad.flat[down])]
            up = up[np.argsort(grad.flat[up])]
            for size in (min(len(up),len(down))//8,min(len(up),len(down))//3):
                if size > 0:
                    m = incumbent.copy()
                    m.flat[down[:size]],m.flat[up[:size]] = 0,1
                    yield m
            if family == "random":
                # Distributed swaps let a dispersed layout improve without
                # concentrating every edit at the same spatial extremum.
                spread = float(grad.std())
                for size in (4,12,32):
                    size = min(size,len(up),len(down))
                    for noise in (.4,1.,2.):
                        scores = grad + rng.normal(0,spread*noise,size=grad.shape)
                        m = incumbent.copy()
                        m.flat[down[np.argsort(-scores.flat[down])[:size]]] = 0
                        m.flat[up[np.argsort(scores.flat[up])[:size]]] = 1
                        yield m
            yield from islice(shape_candidates(family,-grad,incumbent),8,28)
        screen(variations(),2)
        if best[0] == prev:
            break
    if best is None:
        return unavailable("本轮搜索未得到唯一归入所选类别的排布。")
    rank,winning,_,curve,response = best
    actual = response[metric]
    improved = False
    if actual is not None and initial_value is not None:
        delta = (actual-initial_value if target is None else
                 abs(initial_value-target)-abs(actual-target))
        improved = bool(delta > 1e-10*max(1.,abs(initial_value),abs(actual)))
    met = matched() if tol is not None else None
    passed = met if target is not None else improved
    meta.update(curve_Pa=curve.tolist(),response=response,value=actual,improved=improved,
                active=int(winning.sum()),tested_counts=sorted(tested_counts),
                screened_counts=sorted(screened_counts | tested_counts),
                absolute_error=abs(actual-target) if actual is not None and target is not None else None,
                met_target=met,trials=trials,elapsed=perf_counter()-start,
                changed=int(np.count_nonzero(source!=winning)))
    if actual is None:
        reason = "候选在采样频率内没有完整的两侧 −6 dB 交点，带宽无法量化。"
    elif target is not None and tol is None:
        reason = "本轮按与给定值的绝对差选择了最接近的同类候选。"
    elif target is not None:
        reason = "复算结果达到给定值的绝对容差。" if met else "本轮最接近给定值的同类候选仍超出容差。"
    elif initial_value is None:
        reason = "候选具有可测带宽。初图通带未闭合，无法量化增幅。"
    else:
        reason = "本轮最佳同类候选的指标高于初图。" if improved else "本轮最佳同类候选的指标未超过初图。"
    return dict(family=family,status="verified" if passed else "partial",mask=winning.astype(int).tolist(),
                reason=reason,metrics=meta)
