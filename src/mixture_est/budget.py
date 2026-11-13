"""允许拟合损失后的比例范围（loss budget 多面体）。

给定原归一化最大似然 ``L*`` 与非负预算 ``B``，在

    S = { w : w >= 0, sum_j w_j = 1, L(w) >= L* - B }

上求每个类别比例的最小值、最大值及完整 ``w`` 见证。零计数项对似然贡献
零；正计数遇零概率的点 ``L=-inf``，被自动排除在 S 之外。

``L`` 在似然定义域内是凹函数，故 S 是凸紧集，线性目标 min/max w_j 的
最优解落在阈值边界 ``L(w)=t``（``t=L*-B``）上。这里不调用任何优化求解器，
而是手写一套**阈值曲面上的主动集投影梯度**（切平面投影用显式 2x2 求逆，
全部线性运算与二分、梯度均为手写）：

1. 从单纯形目标顶点（min 取 w_j=0 的顶点、max 取 w_j=1 的顶点）出发，沿
   它到 MLE 锚点的线段二分，得到阈值边界上的起点（正计数遇零概率的点
   L=-inf，二分会自动绕开）；
2. 每轮把最速下降方向 ``-c``（c=±e_j）投影到切平面
   ``{d: 1·d=0, grad L·d=0}`` 与活动坐标（w=0）的零空间，得到既保持
   归一化、又一阶保持 L=t 的可行方向；
3. 沿方向走到第一个撞零的坐标，再用线段二分把点拉回阈值边界；
4. 当前活动集无下降方向时，逐个释放活动坐标重启；都无法改善即收敛到
   KKT 最优点。

线性系统（切平面约束的显式 2x2 求逆）与全部二分、梯度均为手写，
不引入第三方库或优化求解器。
"""

import math

from .em import compute_q, log_likelihood

# 主动集外/内层迭代上限
OUTER_ROUNDS = 40
INNER_ROUNDS = 80
# 判定点已落在阈值上的似然容差（与见证收尾一致）
FEAS_L_TOL = 1e-11
# 下降方向判定容差
DESCENT_TOL = 1e-11
# 二分求 L=t 边界点的迭代数（2^-80 远小于机器精度）
BISECT_ROUNDS = 80


def likelihood_gradient(flat_a, offsets, rows, counts, total_n, weights):
    """归一化对数似然的梯度

        grad_j L(w) = (1/N) * sum_ei n_ei A_eij / q_ei(w)

    仅在似然有限（所有正计数单元 q>0）的点调用；零计数项不参与。
    """
    n_classes = len(weights)
    q = compute_q(flat_a, offsets, rows, weights)
    grad = [0.0] * n_classes
    cell = 0
    for e, n_rows in enumerate(rows):
        offset = offsets[e]
        for i in range(n_rows):
            n = counts[offset + i]
            if n:
                qi = q[offset + i]
                if qi <= 0.0:
                    raise ValueError("梯度在零概率点不存在")
                inv = n / (total_n * qi)
                base = cell * n_classes
                for j in range(n_classes):
                    grad[j] += flat_a[base + j] * inv
            cell += 1
    return grad


def _project_tangent(c, g, active, n_classes):
    """把最速下降方向 ``-c`` 投影到切平面零空间。

    求方向 ``d`` 满足 ``sum d=0``、``g·d=0``、活动坐标 ``d_A=0``，并在
    这些约束下最贴近 ``-c``（正交投影到约束零空间）。自由坐标少于 2 个
    时切平面退化，返回零方向。当梯度行与全 1 行在自由坐标上线性相关
    （如单位阵 + 零计数，``g`` 在自由坐标上为常数）时，``g·d=0`` 被
    ``sum d=0`` 蕴含，自动退化为只满足归一化约束的投影。
    """
    free = [k for k in range(n_classes) if k not in active]
    n_free = len(free)
    if n_free < 2:
        return [0.0] * n_classes
    target = [-c[k] for k in free]
    ones = [1.0] * n_free
    grad = [g[k] for k in free]

    def place(projection):
        full = [0.0] * n_classes
        for r, k in enumerate(free):
            full[k] = projection[r]
        return full

    # 只有归一化约束时：d = target - mean(target)·1
    g_centered = [grad[r] - sum(grad) / n_free for r in range(n_free)]
    if sum(v * v for v in g_centered) <= 1e-18:
        mean_t = sum(target) / n_free
        return place([target[r] - mean_t for r in range(n_free)])

    # 两个独立约束：d = target - E^T (E E^T)^-1 E target，显式 2x2 求逆
    e0, e1 = ones, grad
    a00 = sum(x * x for x in e0)              # = n_free
    a01 = sum(e0[r] * e1[r] for r in range(n_free))
    a11 = sum(x * x for x in e1)
    b0 = sum(e0[r] * target[r] for r in range(n_free))
    b1 = sum(e1[r] * target[r] for r in range(n_free))
    det = a00 * a11 - a01 * a01
    if abs(det) <= 1e-18 * max(1.0, a00 * a11):
        # 数值上仍共线：退回单约束投影
        mean_t = sum(target) / n_free
        return place([target[r] - mean_t for r in range(n_free)])
    lam0 = (a11 * b0 - a01 * b1) / det
    lam1 = (a00 * b1 - a01 * b0) / det
    return place([
        target[r] - lam0 * e0[r] - lam1 * e1[r]
        for r in range(n_free)
    ])


def _bisect_to_boundary(vertex, anchor, threshold, flat_a, offsets, rows,
                        counts, total_n):
    """沿 ``vertex -> anchor`` 线段二分，返回最靠近 vertex 的可行点。

    点 ``p(a)=(1-a)*vertex + a*anchor``，锚点满足 ``L>=threshold``、
    vertex 不满足（含 L=-inf）。返回的点满足 ``L(p) ≈ threshold``。
    """
    n = len(vertex)
    lo, hi = 0.0, 1.0
    for _ in range(BISECT_ROUNDS):
        mid = (lo + hi) / 2.0
        point = [(1.0 - mid) * vertex[k] + mid * anchor[k] for k in range(n)]
        value = log_likelihood(
            counts, compute_q(flat_a, offsets, rows, point), total_n
        )
        if math.isfinite(value) and value >= threshold - FEAS_L_TOL:
            hi = mid
        else:
            lo = mid
    return [(1.0 - hi) * vertex[k] + hi * anchor[k] for k in range(n)]


def _clean(point):
    """只清理浮点产生的微小负零；保留一切正分量（哪怕极小）。

    极小正权重（如 1e-15）仍是似然有限的合法点，可能正是零端点见证，
    绝不能截断成 0——那会把它变成正计数遇零概率的不可行点。
    """
    return [0.0 if (v < 0.0 and v > -1e-9) else v for v in point]


def _ensure_finite_likelihood(point, anchor, threshold, flat_a, offsets,
                              rows, counts, total_n):
    """把点修正到似然有限（所有正计数单元 q>0）且满足阈值的位置。

    二分返回的点理论上满足 L>=threshold，但当阈值边界与似然定义域边界
    重合时，可能停在 L=-inf 或概率下溢为 0 的点上，使梯度无法计算。这里
    按几何级数朝锚点收缩，直到似然有限且仍满足阈值。
    """
    def finite_and_feasible(p):
        value = log_likelihood(
            counts, compute_q(flat_a, offsets, rows, p), total_n
        )
        return math.isfinite(value) and value >= threshold - FEAS_L_TOL

    if finite_and_feasible(point):
        return point
    blend = 0.5
    for _ in range(80):
        candidate = [
            (1.0 - blend) * point[k] + blend * anchor[k]
            for k in range(len(point))
        ]
        if finite_and_feasible(candidate):
            return candidate
        blend = 0.5 + 0.5 * blend
    # 锚点本身似然有限，最后直接返回锚点（始终安全）
    return anchor[:]


def _safe_gradient(point, anchor, threshold, flat_a, offsets, rows, counts,
                   total_n):
    """先修正到有限似然点，再计算梯度，返回 (安全点, 梯度)。"""
    safe = _ensure_finite_likelihood(
        point, anchor, threshold, flat_a, offsets, rows, counts, total_n
    )
    return safe, likelihood_gradient(
        flat_a, offsets, rows, counts, total_n, safe
    )


def _renormalize(point):
    """修正投影/二分累积的归一化漂移（仅允许 1e-6 以内的微小修正）。

    算法的每一步都在 sum=1 的仿射平面上，漂移纯粹来自浮点舍入；若漂移
    过大说明逻辑出错，直接抛出而不是掩盖。
    """
    total = sum(point)
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError("budget 见证归一化漂移过大：%.3e" % (total - 1.0))
    if abs(total - 1.0) <= 1e-13:
        return point
    scale = 1.0 / total
    return [max(0.0, v) * scale for v in point]


def _extremum(j, sign, threshold, anchor, target_vertex, flat_a, offsets,
              rows, counts, total_n, n_classes):
    """主动集投影梯度求 w_j 的最小（sign=1）/最大（sign=-1）端点见证。"""
    c = [0.0] * n_classes
    c[j] = sign

    # 1) 线段二分取得阈值边界起点，并保证似然有限（梯度可计算）
    start = _bisect_to_boundary(
        target_vertex, anchor, threshold, flat_a, offsets, rows, counts,
        total_n
    )
    start = _clean(start)
    start = _ensure_finite_likelihood(
        start, anchor, threshold, flat_a, offsets, rows, counts, total_n
    )
    start = _clean(start)
    best = start[:]
    best_obj = start[j] * sign
    # 只把“精确为零”的坐标纳入活动集；二分起点的极小正分量（如 1e-15）
    # 仍是合法的有限似然点，绝不能预先钉死，否则会把真正的零端点排除。
    active = {k for k in range(n_classes) if start[k] == 0.0}

    # 2) 外层：活动集 stall 后释放坐标重启
    for _ in range(OUTER_ROUNDS):
        w = best[:]
        # 3) 内层：固定活动集，沿切平面投影方向走到边界
        for _ in range(INNER_ROUNDS):
            w, g = _safe_gradient(
                w, anchor, threshold, flat_a, offsets, rows, counts, total_n
            )
            direction = _project_tangent(c, g, active, n_classes)
            if sum(c[k] * direction[k] for k in range(n_classes)
                   ) >= -DESCENT_TOL:
                break  # 当前活动集下已无下降方向
            # 走到第一个撞零的自由坐标（步长再截断到 1，保持数值温和）
            alpha_max = math.inf
            for k in range(n_classes):
                if direction[k] < -1e-15:
                    alpha_max = min(alpha_max, w[k] / -direction[k])
            alpha_max = min(alpha_max, 1.0)
            target = [
                w[k] + alpha_max * direction[k] for k in range(n_classes)
            ]
            target = _clean(target)
            # 切向落点可能恰好落在似然定义域边界（正计数遇零概率，
            # L=-inf、梯度不存在）。统一用线段二分到锚点，既拉回阈值
            # 曲面又保证落点似然有限，后续梯度计算始终安全。
            nxt = _bisect_to_boundary(
                target, anchor, threshold, flat_a, offsets, rows, counts,
                total_n
            )
            nxt = _clean(nxt)
            nxt = _ensure_finite_likelihood(
                nxt, anchor, threshold, flat_a, offsets, rows, counts,
                total_n
            )
            obj = nxt[j] * sign
            if obj < best_obj - 1e-12:
                best, best_obj = nxt[:], obj
            if abs(nxt[j] - w[j]) <= 1e-12:
                break
            w = nxt
            # 仅钉死精确零坐标；极小正分量保持自由，以免漏掉零端点
            active = {k for k in range(n_classes) if w[k] == 0.0}

        # 4) 尝试释放一个能重新产生下降方向的活动坐标
        progressed = False
        safe_best, g_best = _safe_gradient(
            best, anchor, threshold, flat_a, offsets, rows, counts, total_n
        )
        best = safe_best
        for k in sorted(active):
            trial_active = set(active)
            trial_active.remove(k)
            direction = _project_tangent(c, g_best, trial_active, n_classes)
            if sum(c[x] * direction[x] for x in range(n_classes)
                   ) < -DESCENT_TOL:
                active = trial_active
                progressed = True
                break
        if not progressed:
            break

    return _finalize(best, threshold, flat_a, offsets, rows, counts, total_n)


def _finalize(witness, threshold, flat_a, offsets, rows, counts, total_n):
    """见证收尾：截断数值噪声、修正归一化漂移并确认似然不低于阈值。"""
    cleaned = _renormalize(_clean(witness))
    value = log_likelihood(
        counts, compute_q(flat_a, offsets, rows, cleaned), total_n
    )
    if not math.isfinite(value) or value < threshold - 1e-8:
        raise RuntimeError(
            "loss budget 见证似然 %.10f 低于阈值 %.10f" % (value, threshold)
        )
    return cleaned


def budget_ranges(classes, flat_a, offsets, rows, counts, total_n,
                  max_l, anchor, budget):
    """计算 loss budget 下每类比例的最小/最大值与完整见证。

    :param budget: 非负 :class:`fractions.Fraction`，允许的似然损失。
    :param anchor: MLE 权重（似然有限的可行点），作为二分回拉的锚点。
    :returns: 与 ``ranges`` 同构的列表，按 ``classes`` 原序。
    """
    n_classes = len(classes)
    threshold = max_l - float(budget)
    result = []
    for j, name in enumerate(classes):
        # 单纯形目标顶点：min 取 w_j=0（质量放到下一个类别），max 取 w_j=1
        min_vertex = [0.0] * n_classes
        min_vertex[(j + 1) % n_classes] = 1.0
        max_vertex = [0.0] * n_classes
        max_vertex[j] = 1.0

        witness_min = _extremum(
            j, 1.0, threshold, anchor, min_vertex, flat_a, offsets, rows,
            counts, total_n, n_classes
        )
        witness_max = _extremum(
            j, -1.0, threshold, anchor, max_vertex, flat_a, offsets, rows,
            counts, total_n, n_classes
        )
        minimum = max(0.0, witness_min[j])
        maximum = min(1.0, witness_max[j])
        if minimum - maximum > 1e-9:
            raise RuntimeError(
                "类别 %s 的 budget 范围端点次序错误" % name
            )
        result.append({
            "class": name,
            "minimum": minimum,
            "maximum": maximum,
            "min_witness": witness_min,
            "max_witness": witness_max,
        })
    return result
