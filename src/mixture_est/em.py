"""EM（乘性迭代）求混合模型最大似然。

目标（忽略常数 1/N）：

    maximize  L(w) = sum_ei n_ei * ln( (A_e w)_i )
    s.t.      w >= 0,  sum_j w_j = 1

EM 的 M 步更新（把缺失的类别归属补全后重新归一化）：

    w_j <- w_j * g_j(w) / N,   g_j(w) = sum_ei n_ei A_eij / q_ei(w)

这是经典的 MM 迭代，保证似然单调不减。主循环内嵌 Type-I Anderson
加速处理相同响应列带来的慢收敛，加速步不提升似然则回退普通 EM。

收敛后用 KKT 驱动的主动集精修区分两类零：
  * 结构性零：g_j < N（目标想把该坐标压到 0），固定为 0；
  * 非可辨识方向：g_j = N（如完全相同的响应列），绝不固定，
    保留给后续最小范数代表解与范围 LP 处理。
"""

import math

from .errors import ImpossibleEstimation
from .linalg import SingularMatrix, solve_linear

KKT_TOL = 1e-9          # g_j 与 N 相等性的判定容差
REFINE_ROUNDS = 8       # pin/unpin 精修最大轮数
EM_FIXED_ROUNDS = 30000


def compute_q(flat_a, offsets, rows, weights):
    n_classes = len(weights)
    q = [0.0] * sum(rows)
    cell = 0
    for n_rows in rows:
        for _ in range(n_rows):
            q[cell] = sum(
                flat_a[cell * n_classes + j] * weights[j]
                for j in range(n_classes)
            )
            cell += 1
    return q


def log_likelihood(counts, q, total_n):
    """归一化对数似然 L = sum n ln q / N；正计数遇零概率返回 -inf。"""
    total = 0.0
    for n, qi in zip(counts, q):
        if n == 0:
            continue
        if qi <= 0.0:
            return -math.inf
        total += n * math.log(qi)
    return total / total_n


def gradient_g(flat_a, offsets, rows, counts, q, n_classes):
    """计算 g_j = sum_ei n_ei A_eij / q_ei。正计数遇零概率抛异常。"""
    g = [0.0] * n_classes
    cell = 0
    for e, n_rows in enumerate(rows):
        for i in range(n_rows):
            n = counts[offsets[e] + i]
            if n:
                qi = q[offsets[e] + i]
                if qi <= 0.0:
                    raise ImpossibleEstimation("正计数遇到零概率")
                inv = n / qi
                base = cell * n_classes
                for j in range(n_classes):
                    g[j] += flat_a[base + j] * inv
            cell += 1
    return g


def _anderson_step(history_g, history_f, n_classes):
    """Type-I Anderson 加速。失败返回 None，调用方回退普通 EM。"""
    depth = len(history_g)
    if depth < 2:
        return None
    gram = [[0.0] * depth for _ in range(depth)]
    rhs = [0.0] * depth
    latest_g = history_g[-1]
    for a in range(depth):
        for b in range(depth):
            gram[a][b] = sum(
                history_g[a][j] * history_g[b][j] for j in range(n_classes)
            )
        rhs[a] = sum(history_g[a][j] * latest_g[j] for j in range(n_classes))

    scale = max(1.0, max(abs(v) for row in gram for v in row))
    gamma = None
    for attempt in range(4):
        try:
            gamma = solve_linear(gram, rhs,
                                 ridge=1e-12 * scale * (10 ** attempt))
            break
        except SingularMatrix:
            continue
    if gamma is None:
        return None

    gamma = [min(1.0e4, max(-1.0e4, value)) for value in gamma]
    denom = sum(gamma)
    if abs(denom) < 1e-12:
        return None

    candidate = [0.0] * n_classes
    for j in range(n_classes):
        w_mix = sum(gamma[k] * history_f[k][j] for k in range(depth)) / denom
        g_mix = sum(gamma[k] * history_g[k][j] for k in range(depth)) / denom
        value = w_mix * g_mix
        if value <= 0.0:
            return None
        candidate[j] = value
    norm = sum(candidate)
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return [value / norm for value in candidate]


def _em_on_free(flat_a, offsets, rows, counts, total_n, n_classes, fixed_zero):
    """在固定部分坐标为零的面上跑 EM，返回 (权重, q, 似然)。"""
    free = [j for j in range(n_classes) if j not in fixed_zero]
    weights = [0.0] * n_classes
    for j in free:
        weights[j] = 1.0 / len(free)
    q = compute_q(flat_a, offsets, rows, weights)
    prev_l = log_likelihood(counts, q, total_n)
    if not math.isfinite(prev_l):
        return weights, q, prev_l

    for _ in range(EM_FIXED_ROUNDS):
        g = gradient_g(flat_a, offsets, rows, counts, q, n_classes)
        denom = sum(weights[j] * g[j] for j in free)
        if denom <= 0.0:
            break
        new_weights = [0.0] * n_classes
        for j in free:
            new_weights[j] = weights[j] * g[j] / denom
        new_q = compute_q(flat_a, offsets, rows, new_weights)
        new_l = log_likelihood(counts, new_q, total_n)
        if not math.isfinite(new_l):
            break
        change = max(
            abs(new_weights[j] - weights[j]) for j in free
        )
        weights, q = new_weights, new_q
        if change < 1e-14 or abs(new_l - prev_l) <= 1e-15 * max(
                1.0, abs(prev_l)):
            prev_l = new_l
            break
        prev_l = new_l
    return weights, q, prev_l


def fit_mle(flat_a, offsets, rows, counts, total_n, n_classes, max_iter=300000):
    """返回 MLE 权重（浮点列表）。调用方已保证问题可行。"""
    if n_classes == 1:
        # 单类别时归一化约束唯一确定 w=1，无需迭代
        return [1.0]
    weights = [1.0 / n_classes] * n_classes
    q = compute_q(flat_a, offsets, rows, weights)
    best_l = log_likelihood(counts, q, total_n)

    history_f = []
    history_g = []
    for iteration in range(max_iter):
        g = gradient_g(flat_a, offsets, rows, counts, q, n_classes)
        plain = [weights[j] * g[j] / total_n for j in range(n_classes)]
        history_f.append(weights[:])
        history_g.append(g[:])
        if len(history_f) > 6:
            history_f.pop(0)
            history_g.pop(0)

        candidate = plain
        if iteration >= 1 and iteration % 8 == 0:
            accelerated = _anderson_step(history_g, history_f, n_classes)
            if accelerated is not None:
                q_acc = compute_q(flat_a, offsets, rows, accelerated)
                if all(qi > 0.0 for qi in q_acc):
                    l_acc = log_likelihood(counts, q_acc, total_n)
                    if math.isfinite(l_acc) and l_acc >= best_l - 1e-13:
                        candidate = accelerated

        new_q = compute_q(flat_a, offsets, rows, candidate)
        new_l = log_likelihood(counts, new_q, total_n)
        if not math.isfinite(new_l) or new_l + 1e-14 < best_l:
            candidate = plain
            new_q = compute_q(flat_a, offsets, rows, candidate)
            new_l = log_likelihood(counts, new_q, total_n)

        change = max(
            abs(candidate[j] - weights[j]) / max(1.0, abs(weights[j]))
            for j in range(n_classes)
        )
        weights, q, best_l = candidate, new_q, new_l
        if change < 1e-14:
            break

    # ---- KKT 驱动的主动集精修 ----
    # 主 EM 可能收敛到 MLE 多面体的某个顶点，把非可辨识方向也压在零上。
    # 这里从“全部坐标自由”的内部点重新出发，按 KKT 乘子符号逐个加入
    # 固定坐标，确保只有真正的结构性零（g_j<N）被固定。
    fixed = set()
    weights, q, best_l = _em_on_free(
        flat_a, offsets, rows, counts, total_n, n_classes, fixed
    )

    for _ in range(REFINE_ROUNDS):
        g = gradient_g(flat_a, offsets, rows, counts, q, n_classes)
        # 结构性零候选：自由坐标、目标主动想压零（g_j < N）。
        # 不要求坐标已经极小——边界 MLE 的 EM 可能停在 1e-5 量级，
        # 用 KKT 乘子符号判定后固定，再在面上重跑即可精确归零。
        candidates = [
            j for j in range(n_classes)
            if j not in fixed
            and g[j] < total_n * (1.0 - KKT_TOL)
        ]
        if not candidates:
            break
        # 优先处理“乘子差距最大、且当前坐标最小”的坐标
        j_star = min(candidates,
                     key=lambda j: (g[j] / total_n, weights[j]))
        trial = set(fixed)
        trial.add(j_star)
        refined, q_ref, l_ref = _em_on_free(
            flat_a, offsets, rows, counts, total_n, n_classes, trial
        )
        if not math.isfinite(l_ref) or l_ref < best_l - 1e-9:
            # 钉死它会损失似然 -> 不是结构性零，标记后不再考虑
            g[j_star] = total_n  # 屏蔽该候选
            continue
        # 复查被固定坐标的 KKT 条件 g_j <= N
        g_ref = gradient_g(flat_a, offsets, rows, counts, q_ref, n_classes)
        if g_ref[j_star] > total_n * (1.0 + KKT_TOL):
            g[j_star] = total_n
            continue
        fixed.add(j_star)
        weights, q, best_l = refined, q_ref, l_ref

    # 数值清理；仅在归一化确实漂移时才重新归一化，避免无谓的浮点噪声
    weights = [0.0 if abs(w) < 1e-12 else w for w in weights]
    norm = sum(weights)
    if abs(norm - 1.0) > 1e-13:
        return [w / norm for w in weights]
    return weights
