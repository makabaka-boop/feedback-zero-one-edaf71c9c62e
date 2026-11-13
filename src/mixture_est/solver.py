"""主编排：可行性判定、MLE、最小范数代表解、比例范围与见证。"""

import math

from .em import compute_q, fit_mle, log_likelihood
from .errors import ImpossibleEstimation
from .polytope import linear_program, min_norm

OUT_TOL = 1e-6
L_TOL = 1e-8


def _prepare(validated):
    classes = validated["classes"]
    n_classes = len(classes)
    flat_a = []
    offsets = []
    rows = []
    counts = []
    offset = 0
    for experiment in validated["experiments"]:
        offsets.append(offset)
        matrix = experiment["A"]
        n_rows = len(matrix)
        rows.append(n_rows)
        for row in matrix:
            flat_a.extend(float(value) for value in row)
        counts.extend(experiment["n"])
        offset += n_rows
    total_n = sum(counts)
    return classes, n_classes, flat_a, offsets, rows, counts, total_n


def _is_impossible(flat_a, offsets, rows, counts, n_classes):
    """正计数单元在某实验中对应整行为零，则没有任何可行权重能给出正概率。"""
    cell = 0
    for e, offset in enumerate(offsets):
        for i in range(rows[e]):
            if counts[offset + i] > 0:
                base = cell * n_classes
                if all(flat_a[base + j] == 0.0 for j in range(n_classes)):
                    return True
            cell += 1
    return False


def _build_polytope(flat_a, offsets, rows, q_star, weights):
    """构造 MLE 集合 P = {w : sum w=1, Mw=q*, w>=0} 的等式约束。

    去掉与既有约束线性冗余（数值意义下）的行，避免单纯形退化。
    """
    n_classes = len(weights)

    # 收集 MLE 等式 + 归一化行。
    # q*=0 的行 A_row·w=0 在 w>=0、A>=0 下等价于“凡 A_rowj>0 的类别 w_j=0”，
    # 该蕴含已被非负约束与其余 q*>0 等式在 MLE 面上自动保证，故跳过，
    # 避免 EM 残留小值让 RREF 误判约束矛盾。
    candidates = [([1.0] * n_classes, 1.0)]
    cell = 0
    for e, offset in enumerate(offsets):
        for i in range(rows[e]):
            base = cell * n_classes
            row = flat_a[base:base + n_classes]
            rhs = q_star[offset + i]
            if rhs > 1e-10:
                candidates.append((row, rhs))
            cell += 1

    # 对增广矩阵 [C | d] 做 RREF 同时保留系数与 RHS 的线性关系。
    # 只按系数秩选行会丢失“冗余系数行对应的 RHS 组合”这一信息，
    # 因此这里直接取增广 RREF 的主元行作为独立等式。
    # q* 由可行的 EM 权重生成，非主元行消元后的 RHS 必为浮点级的 0。
    aug = [list(row) + [rhs] for row, rhs in candidates]
    pivot_rows = _rref(aug, n_classes)
    eq_rows, eq_rhs = [], []
    for r in pivot_rows:
        row = aug[r][:n_classes]
        rhs_value = aug[r][n_classes]
        # 两阶段单纯形要求 RHS 非负以构造人工初始基；RREF 可能给出负 RHS，
        # 整行乘 -1 不改变等式。
        if rhs_value < -1e-12:
            row = [-v for v in row]
            rhs_value = -rhs_value
        eq_rows.append(row)
        eq_rhs.append(rhs_value)
    return eq_rows, eq_rhs


def _rref(augmented, n_cols):
    """就地行约化（Gauss-Jordan），返回主元行下标列表。"""
    n_rows = len(augmented)
    pivot_rows = []
    row_idx = 0
    scale = max(
        (abs(augmented[r][c])
         for r in range(n_rows) for c in range(n_cols)),
        default=1.0,
    )
    tol = 1e-11 * max(1.0, scale)
    for col in range(n_cols):
        pivot = max(
            range(row_idx, n_rows),
            key=lambda r: abs(augmented[r][col]),
            default=None,
        )
        if pivot is None or abs(augmented[pivot][col]) <= tol:
            continue
        augmented[row_idx], augmented[pivot] = augmented[pivot], augmented[row_idx]
        pivot_value = augmented[row_idx][col]
        augmented[row_idx] = [v / pivot_value for v in augmented[row_idx]]
        for r in range(n_rows):
            if r == row_idx:
                continue
            factor = augmented[r][col]
            if factor:
                augmented[r] = [
                    augmented[r][c] - factor * augmented[row_idx][c]
                    for c in range(n_cols + 1)
                ]
        pivot_rows.append(row_idx)
        row_idx += 1
        if row_idx == n_rows:
            break
    return pivot_rows


def _clean_point(point):
    # 只截断数值噪声产生的微小负值/零，绝不重新归一化：
    # MLE 等式包含非单纯形行，归一化会破坏这些约束。
    cleaned = []
    for v in point:
        if abs(v) < 1e-10:
            cleaned.append(0.0)
        elif v < 0.0 and v > -1e-9:
            cleaned.append(0.0)
        else:
            cleaned.append(v)
    return cleaned


def _check_feasible(point, eq_rows, eq_rhs, label):
    for i, row in enumerate(eq_rows):
        residual = eq_rhs[i] - sum(row[j] * point[j] for j in range(len(point)))
        if abs(residual) > 1e-7:
            raise RuntimeError(
                "%s 不满足 MLE 等式约束（残差 %.3e）" % (label, residual)
            )
    if any(v < -1e-9 for v in point):
        raise RuntimeError("%s 出现负分量" % label)


def estimate(validated):
    """主入口。返回可 JSON 序列化的结果字典；不可行时抛 :class:`ImpossibleEstimation`。"""
    (
        classes, n_classes, flat_a, offsets, rows, counts, total_n
    ) = _prepare(validated)

    if _is_impossible(flat_a, offsets, rows, counts, n_classes):
        raise ImpossibleEstimation(
            "存在正计数落在全零响应行，所有可行权重的似然均为负无穷"
        )

    # 1) EM 求一个 MLE 权重，进而确定唯一的拟合概率 q*
    weights = fit_mle(flat_a, offsets, rows, counts, total_n, n_classes)
    q_star = compute_q(flat_a, offsets, rows, weights)
    max_l = log_likelihood(counts, q_star, total_n)
    if not math.isfinite(max_l):
        raise ImpossibleEstimation("内部错误：EM 未找到有限似然解")

    # 2) MLE 集合的等式约束多面体
    eq_rows, eq_rhs = _build_polytope(flat_a, offsets, rows, q_star, weights)

    # 3) 所有 MLE 解中 sum w_j^2 最小的唯一代表
    representative = min_norm(eq_rows, eq_rhs, weights)
    representative = _clean_point(representative)
    _check_feasible(representative, eq_rows, eq_rhs, "代表解")

    # 4) 每个类别的比例最小/最大值及见证（单纯形 LP，顶点解）
    ranges = []
    for j in range(n_classes):
        c_min = [0.0] * n_classes
        c_min[j] = 1.0
        witness_min = linear_program(c_min, eq_rows, eq_rhs, weights)
        witness_min = _clean_point(witness_min)
        _check_feasible(witness_min, eq_rows, eq_rhs,
                        "类别 %s 的最小端点见证" % classes[j])

        c_max = [0.0] * n_classes
        c_max[j] = -1.0
        witness_max = linear_program(c_max, eq_rows, eq_rhs, weights)
        witness_max = _clean_point(witness_max)
        _check_feasible(witness_max, eq_rows, eq_rhs,
                        "类别 %s 的最大端点见证" % classes[j])

        minimum = max(0.0, witness_min[j])
        maximum = min(1.0, witness_max[j])
        if minimum - maximum > 1e-9:
            raise RuntimeError("类别 %s 的范围端点次序错误" % classes[j])
        ranges.append({
            "class": classes[j],
            "minimum": minimum,
            "maximum": maximum,
            "min_witness": witness_min,
            "max_witness": witness_max,
        })

    # 5) 各实验拟合概率（按实验原序、观测行原序）
    fitted_probabilities = []
    for e, experiment in enumerate(validated["experiments"]):
        start = offsets[e]
        fitted_probabilities.append({
            "id": experiment["id"],
            "q": q_star[start:start + rows[e]],
        })

    # 6) 自洽性核对：见证与代表必须达到同一归一化似然（容差 1e-8）
    for point in [representative] + [
        w for r in ranges for w in (r["min_witness"], r["max_witness"])
    ]:
        q_check = compute_q(flat_a, offsets, rows, point)
        l_check = log_likelihood(counts, q_check, total_n)
        if abs(l_check - max_l) > L_TOL:
            raise RuntimeError(
                "端点见证似然不一致：%.10f vs %.10f" % (l_check, max_l)
            )

    # 7) 有限 JSON 数核对
    def finite(values):
        return all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in values)

    if not finite(representative) or not finite(q_star):
        raise RuntimeError("输出包含非有限数")
    for r in ranges:
        if not finite(r["min_witness"]) or not finite(r["max_witness"]):
            raise RuntimeError("见证包含非有限数")

    return {
        "status": "ok",
        "classes": classes,
        "N": total_n,
        "log_likelihood": max_l,
        "representative": representative,
        "ranges": ranges,
        "fitted_probabilities": fitted_probabilities,
    }
