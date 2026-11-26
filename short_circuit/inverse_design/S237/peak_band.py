"""Fixed-count PMUT design at the target peak and its two −6 dB crossings."""
from itertools import islice
from functools import lru_cache
from time import perf_counter

from family_design import (FAMILIES, FREQ, G, N, FB, certified, conv, make_mask,
                           mask_array, np, proposals, roi, starting_field)
from common import L


LEVEL = 10.0 ** (-6.0 / 20.0)
CURVE_KEYS = ("peak_Pa", "peak_MHz", "left_MHz", "right_MHz", "relative_bandwidth_pct")
PLANE_KEYS = ("peak_plane_pct", "left_plane_pct", "right_plane_pct")
KEY_FREQUENCIES = ("peak_MHz", "left_MHz", "right_MHz")


def features(curve, frequencies=FREQ):
    """First global sampled peak, then the nearest crossing on each side."""
    y = np.asarray(curve, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if (y.ndim != 1 or y.size == 0 or y.shape != f.shape or not np.isfinite(y).all()
            or not np.isfinite(f).all() or np.any(np.diff(f) <= 0) or np.any(y < 0)):
        raise ValueError("频响须包含递增频率和有限、非负的幅值。")
    i = int(np.argmax(y))
    peak = float(y[i])
    threshold = peak * LEVEL
    values = dict(peak_Pa=peak, peak_MHz=None, threshold_Pa=threshold,
                  left_MHz=None, right_MHz=None, bandwidth_MHz=None, relative_bandwidth_pct=None)
    if peak == 0:
        return values
    values["peak_MHz"] = float(f[i])
    for step, key in ((-1,"left_MHz"), (1,"right_MHz")):
        for j in range(i+step, -1 if step < 0 else len(f), step):
            if y[j] <= threshold:
                near = j-step
                fraction = (threshold-y[j])/(y[near]-y[j])
                values[key] = float(f[j]+fraction*(f[near]-f[j]))
                break
    left, right = values["left_MHz"], values["right_MHz"]
    if left is not None and right is not None:
        values["bandwidth_MHz"] = right-left
        values["relative_bandwidth_pct"] = 200*(right-left)/(right+left)
    return values


def spectrum(pressure, scope, point):
    if scope == "plane":
        return abs(pressure).mean(axis=(1,2))
    if scope == "point":
        if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                any(type(i) is not int or not 0 <= i < 50 for i in point)):
            raise ValueError("所选测点需要两个 0–49 的整数行列索引。")
        return abs(pressure[:,point[0],point[1]])
    if scope == "center":
        return abs(pressure[:,24:26,24:26].mean(axis=(1,2)))
    if scope == "roi":
        return abs(roi(pressure)).mean(axis=(1,2))
    raise ValueError("频响位置须为整面平均、几何中心、中心25点平均或所选测点。")


def weights_at(frequency):
    hi = int(np.searchsorted(FREQ, frequency))
    if FREQ[hi] == frequency:
        return [(hi,1.)]
    lo = hi-1
    t = float((frequency-FREQ[lo])/(FREQ[hi]-FREQ[lo]))
    return [(lo,1-t),(hi,t)]


def amplitude_plane(field, weights):
    return sum(weight*abs(field[k]) for k,weight in weights)


def phase(field):
    magnitude = abs(field)
    return np.divide(field,magnitude,out=np.zeros_like(field),where=magnitude>0)


def curve_receiver(field,scope,point):
    """Adjoint of the selected spatial observation, with phase at the sampled point."""
    if scope == "plane":
        return phase(field)/field.size
    if scope == "roi":
        return L.T @ phase(roi(field)) @ L / 25
    receiver = np.zeros_like(field)
    if scope == "center":
        receiver[24:26,24:26] = phase(field[24:26,24:26].mean())/4
    else:
        row,col = point
        receiver[row,col] = phase(field[row,col])
    return receiver


@lru_cache(maxsize=2)
def peak_dictionary(k):
    """Whiten spatial modes of individual driven cells at this target's peak."""
    from cluster_transfer import table, _rotate
    basis,canonical,orientations,bands = table()
    d = min(19,bands[k].shape[-1])
    band = bands[k][:,:d,:d]
    coefficients = np.empty((2500,d,d),dtype=np.complex64)
    signs = (-1.)**np.arange(d)
    for orientation in range(8):
        ids = np.flatnonzero(orientations==orientation)
        coefficients[ids] = _rotate(band[canonical[ids]],orientation,signs)
    rms = np.sqrt((abs(coefficients)**2).mean(axis=0))
    usable = rms > rms.max()*1e-5
    scaled = coefficients[:,usable]/rms[usable]
    rows = np.column_stack((scaled.real,scaled.imag)).astype(float)
    eigen,vec = np.linalg.eigh(rows.T@rows)
    keep = eigen > eigen[-1]*1e-7
    transform = vec[:,keep]/np.sqrt(eigen[keep])
    atoms = np.ascontiguousarray(rows@transform,dtype=np.float32)
    return basis[:,:d],usable,rms[usable],transform,atoms,np.einsum("ij,ij->i",atoms,atoms)


def infer_at_peak(target,peak,count):
    """Fixed-count matching pursuit from the complex target peak plane."""
    k = int(np.searchsorted(FREQ,peak))
    basis,usable,rms,transform,atoms,norm = peak_dictionary(k)
    observation = (basis.T@target[k]@basis)[usable]/rms
    residual = (np.r_[observation.real,observation.imag]@transform).astype(np.float32)
    if count>1250:
        residual = atoms.sum(axis=0)-residual
    chosen = np.zeros(2500,dtype=bool)
    def place():
        gain = 2*(atoms@residual)-norm
        gain[chosen] = -np.inf
        j = int(gain.argmax())
        residual[:] -= atoms[j]
        chosen[j] = True
        return j
    ids = [place() for _ in range(min(count,2500-count))]
    for _ in range(14):
        previous = chosen.copy()
        for i,j in enumerate(ids):
            residual += atoms[j]
            chosen[j] = False
            ids[i] = place()
        if np.array_equal(previous,chosen):
            break
    return (chosen if count<=1250 else ~chosen).reshape(50,50).astype(float)


class PeakBandObjective:
    def __init__(self,target,scope,point):
        self.target = np.asarray(target)
        if self.target.shape != (len(FREQ),50,50) or not np.isfinite(self.target).all():
            raise ValueError("目标必须为 23×50×50 的有限复声场。")
        self.scope, self.point = scope, point
        self.reference = features(spectrum(self.target,scope,point))
        self.maps = {}
        for key,frequency in zip(PLANE_KEYS,KEY_FREQUENCIES):
            f = self.reference[frequency]
            if f is not None:
                weights = weights_at(f)
                plane = amplitude_plane(self.target,weights)
                norm = float(np.linalg.norm(plane))
                if not np.isfinite(norm):
                    raise ValueError("目标声压过大，无法计算有限的误差。")
                self.maps[key] = weights,plane,norm
        # Scales only balance ranks across unlike units. Acceptance compares
        # the eight errors with their own starting values in their native units.
        self.scales = dict(peak_Pa=max(self.reference["peak_Pa"],1e-12),
                           peak_MHz=1.,left_MHz=1.,right_MHz=1.,
                           relative_bandwidth_pct=100.,**dict.fromkeys(PLANE_KEYS,100.))

    def measure(self,field):
        response = features(spectrum(field,self.scope,self.point))
        errors = {}
        for key in CURVE_KEYS:
            a,b = response[key],self.reference[key]
            errors[key] = abs(a-b) if a is not None and b is not None else None
        for key in PLANE_KEYS:
            errors[key] = None
            if key in self.maps:
                weights,goal,norm = self.maps[key]
                if norm > 0:
                    errors[key] = float(100*np.linalg.norm(amplitude_plane(field,weights)-goal)/norm)
        return response,errors

    def compare(self,errors,baseline):
        checks = {}
        improved = False
        exact = True
        missing = 0
        overshoot = 0.
        total = 0.
        for key,value in errors.items():
            previous = baseline[key]
            if value is None:
                target_key = KEY_FREQUENCIES[PLANE_KEYS.index(key)] if key in PLANE_KEYS else key
                checks[key] = None if self.reference[target_key] is None else False
                exact = False
                missing += 1
                continue
            scale = self.scales[key]
            tolerance = 1e-10*max(scale, value, previous or 0.)
            checks[key] = bool(previous is None or value <= previous+tolerance)
            improved |= previous is None or value < previous-tolerance
            exact &= value <= tolerance
            total += value/scale
            if previous is not None:
                overshoot += max(0.,value-previous-tolerance)/scale
        passed = all(x is True for x in checks.values()) and (improved or exact)
        rank = (not passed, missing, sum(x is False for x in checks.values()), overshoot,total)
        return rank, checks, bool(passed), bool(exact)

    def gradient(self,mask,field,flow):
        # Use local slope information for proposals. The actual search rank
        # always uses the candidates' re-extracted peak and both crossings.
        from protected_linear import solve
        receivers = {}
        for weights,goal,norm in self.maps.values():
            if norm == 0:
                continue
            difference = (amplitude_plane(field,weights)-goal)/norm**2
            for i,w in weights:
                receivers[i] = receivers.get(i,0.) + w*difference*phase(field[i])

        curve = spectrum(field,self.scope,self.point)
        peak = self.reference["peak_Pa"]
        if peak > 0:
            own = int(np.argmax(curve))
            main = self.maps["peak_plane_pct"][0][0][0]
            coefficients = np.zeros(len(curve))
            coefficients[own] += (curve[own]-peak)/peak**2
            if main != own:
                # Encourage the target peak bin to catch up to the current peak.
                gap = (curve[own]-curve[main])/peak**2
                coefficients[own] += gap
                coefficients[main] -= gap
            for name in ("left_plane_pct","right_plane_pct"):
                if name not in self.maps:
                    continue
                weights = self.maps[name][0]
                band_value = sum(curve[k]*w for k,w in weights)
                difference = (band_value-LEVEL*curve[own])/peak**2
                coefficients[own] -= LEVEL*difference
                for k,w in weights:
                    coefficients[k] += w*difference
            for i in np.flatnonzero(coefficients):
                receivers[i] = receivers.get(i,0.) + coefficients[i]*curve_receiver(
                    field[i],self.scope,self.point)
        grad = np.zeros(mask.shape,dtype=float)
        for i,receiver in receivers.items():
            if np.any(receiver):
                adj = solve(mask,i,conv(G[i].conj(),receiver),True)
                grad += np.real(adj.conj()*(N[i]+FB[i]*flow[i]))
        return grad


def design_peak_band(mask,family,target,*,scope="plane",point=None):
    """Search only uniquely certified layouts and return a native-field replay."""
    from cluster_transfer import field as estimate
    from geometry_screen import shape_candidates
    source = mask_array(mask)
    count = int(source.sum())
    if family not in FAMILIES:
        raise ValueError("请选择支持的阵列类别。")
    obj = PeakBandObjective(target,scope,point)
    t0 = perf_counter()
    deadline = t0+12.
    common = dict(scope=scope,point=point if scope == "point" else None,
                  active=count,initial_active=count,target_response=obj.reference)
    def unavailable(reason):
        return dict(family=family,status="unavailable",mask=None,reason=reason,metrics=common)
    if not 6 <= count <= 2494:
        return unavailable("该固定通电数超出原五类唯一分类的有效范围 6–2494。")

    ks = tuple(range(len(FREQ)))
    initial,initial_flow = starting_field(source,ks)
    first,baseline = obj.measure(initial)
    rng = np.random.default_rng(61701+FAMILIES.index(family))
    best = None
    trials = 0
    seen = set()
    def record(candidate,waves):
        nonlocal best,trials
        field,flow = waves
        response,errors = obj.measure(field)
        rank,checks,passed,exact = obj.compare(errors,baseline)
        trials += 1
        if best is None or rank < best[0]:
            best = rank,candidate.copy(),field,flow,response,errors,checks,passed,exact

    def finalists(candidates,limit):
        short = []
        for candidate in candidates:
            if perf_counter() >= deadline:
                break
            if int(candidate.sum()) != count:
                continue
            token = np.packbits(candidate.astype(bool)).tobytes()
            if token in seen:
                continue
            seen.add(token)
            if not certified(candidate,family):
                continue
            if best is None:
                record(candidate,starting_field(candidate,ks))
                if best[7] or not obj.maps:
                    break
                continue
            _,e = obj.measure(estimate(candidate))
            short.append((obj.compare(e,baseline)[0],candidate))
        for _,candidate in sorted(short,key=lambda pair:pair[0])[:limit]:
            if perf_counter() >= deadline or best[7]:
                break
            record(candidate,starting_field(candidate,ks))

    if certified(source,family):
        seen.add(np.packbits(source.astype(bool)).tobytes())
        record(source,(initial,initial_flow))
    if (min(count,2500-count) <= 24 and obj.reference["peak_MHz"] is not None
            and (best is None or not best[7]) and perf_counter() < deadline):
        recovered = infer_at_peak(obj.target,obj.reference["peak_MHz"],count)
        token = np.packbits(recovered.astype(bool)).tobytes()
        if token not in seen and int(recovered.sum()) == count and certified(recovered,family):
            seen.add(token)
            record(recovered,starting_field(recovered,ks))
    if best is None or (not best[7] and obj.maps):
        # Geometry seeds sample layouts beyond the local basin of the input.
        # D4 rotations add spatially distinct maps even for the same aperture.
        def starters():
            for turns in (1,2,3):
                yield np.rot90(source,turns)
            for _ in range(24):
                yield make_mask(rng,count,family)[0].astype(float)
        finalists(starters(),3)
    for _ in range(3):
        if best is None or best[7] or perf_counter() >= deadline or not obj.maps:
            break
        old_rank,incumbent,wave,flow = best[:4]
        grad = obj.gradient(incumbent,wave,flow)
        def variations():
            yield from proposals(incumbent,grad)
            yield from islice(shape_candidates(family,-grad,incumbent),24)
            for _ in range(12):
                changed = incumbent.copy()
                moved = min(count,2500-count,int(rng.integers(1,8)))
                changed.flat[rng.choice(np.flatnonzero(incumbent),moved,replace=False)] = 0
                changed.flat[rng.choice(np.flatnonzero(1-incumbent),moved,replace=False)] = 1
                yield changed
        finalists(variations(),3)
        if best[0] == old_rank:
            break
    if best is None:
        return unavailable("本轮限额内未找到该类别的唯一分类排布。")
    _,winning,wave,_,response,errors,checks,passed,exact = best
    complete = all(v["bandwidth_MHz"] is not None for v in (response,obj.reference))
    ref_norm = np.linalg.norm(obj.target)
    delta = np.linalg.norm(wave-obj.target)
    exact_field = bool(delta <= ref_norm*1e-11)
    metrics = dict(**common,initial_response=first,response=response,errors_before=baseline,
                   errors_after=errors,checks=checks,complete_band=complete,protected_ok=passed,
                   exact_response=exact,exact_field=exact_field,trials=trials,
                   changed=int((source!=winning).sum()),elapsed=perf_counter()-t0)
    if passed:
        reason = ("峰值、两侧带边及三个目标频率的整面幅值图均已数值匹配。" if exact else
                  "相对初图，八项误差均受保护，且至少一项改善或由缺失变为可测。")
    elif obj.reference["peak_Pa"] == 0:
        reason = "目标频响为零，无法确定峰频与 −6 dB 带边。"
    elif not complete:
        reason = "目标或候选的 −6 dB 通带在采样范围内未闭合，已返回可测指标。"
    else:
        reason = "该分类成立，本轮尚未使峰值、两侧带边及三幅整面误差同时满足条件。"
    return dict(family=family,status="verified" if passed else "partial",
                mask=winning.astype(int).tolist(),metrics=metrics,reason=reason)
