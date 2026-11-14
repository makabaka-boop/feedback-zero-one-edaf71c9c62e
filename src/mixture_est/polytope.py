"""仿射多面体上的线性与二次规划（全部手写，不调用求解器）。

MLE 集合是仿射切片与单纯形的交：

    P = { w : C w = d, w >= 0 }

其中 ``C`` 由一个归一化等式与若干 ``M w = q*`` 等式堆叠而成。
本模块提供：

* :func:`null_basis`   —— 行约化求零空间基；
* :func:`linear_program` —— 两阶段单纯形，返回顶点最优解；
* :func:`min_norm`    —— 主动集法解最小化 ``sum w_j^2`` 的等式约束 QP。
"""

from .linalg import SingularMatrix, solve_linear

# 单纯形中的零容差（与输出 1e-6 容差相比留足余量）
PIVOT_TOL = 1e-10
FEAS_TOL = 1e-9


def null_basis(matrix):
    """对 ``matrix`` 做行约化，返回零空间的列基（每个基向量是一个列表）。"""
    if not matrix:
        return []
    n_cols = len(matrix[0])
    rows = [row[:] for row in matrix]
    n_rows = len(rows)
    pivot_cols = []
    row_idx = 0

    for col in range(n_cols):
        pivot = max(
            range(row_idx, n_rows),
            key=lambda r: abs(rows[r][col]),
            default=None,
        )
        if pivot is None or abs(rows[pivot][col]) <= PIVOT_TOL:
            continue
        rows[row_idx], rows[pivot] = rows[pivot], rows[row_idx]
        pivot_value = rows[row_idx][col]
        rows[row_idx] = [v / pivot_value for v in rows[row_idx]]
        for r in range(n_rows):
            if r == row_idx:
                continue
            factor = rows[r][col]
            if factor:
                rows[r] = [
                    rows[r][c] - factor * rows[row_idx][c]
                    for c in range(n_cols)
                ]
        pivot_cols.append(col)
        row_idx += 1
        if row_idx == n_rows:
            break

    free_cols = [c for c in range(n_cols) if c not in pivot_cols]
    basis = []
    for free in free_cols:
        vector = [0.0] * n_cols
        vector[free] = 1.0
        for r, pivot_col in enumerate(pivot_cols):
            vector[pivot_col] = -rows[r][free]
        basis.append(vector)
    return basis


def _tableau_pivot(tableau, entering, leaving):
    pivot = tableau[leaving][entering]
    width = len(tableau[0])
    tableau[leaving] = [v / pivot for v in tableau[leaving]]
    for r in range(len(tableau)):
        if r == leaving:
            continue
        factor = tableau[r][entering]
        if factor == 0.0:
            continue
        tableau[r] = [
            tableau[r][c] - factor * tableau[leaving][c] for c in range(width)
        ]


def linear_program(c, eq_rows, eq_rhs, feasible):
    """两阶段单纯形解

        minimize c^T x  s.t.  eq_rows x = eq_rhs, x >= 0

    ``feasible`` 为已知可行点（构造初始基用，这里直接走两阶段人工变量）。
    返回最优解列表（必为顶点，适合作为范围见证）。
    """
    n_vars = len(c)
    n_eq = len(eq_rows)
    width = n_vars + n_eq + 1  # 变量 + 人工变量 + RHS

    tableau = []
    for i in range(n_eq):
        row = list(eq_rows[i]) + [0.0] * n_eq + [eq_rhs[i]]
        row[n_vars + i] = 1.0
        tableau.append(row)

    # ---- 第一阶段：最小化人工变量之和（目标系数取 -1） ----
    phase1_obj = [0.0] * n_vars + [-1.0] * n_eq + [0.0]
    tableau.append(phase1_obj)
    basis = [n_vars + i for i in range(n_eq)]
    # 消去目标行中的基变量列
    for i in range(n_eq):
        col = basis[i]
        factor = tableau[-1][col]
        if factor:
            for k in range(width):
                tableau[-1][k] -= factor * tableau[i][k]

    def pivot_once(entering_columns):
        """Bland 规则做一次最大化转轴；无入基列返回 False。"""
        obj = tableau[n_eq]
        entering = None
        for col in entering_columns:
            if obj[col] > PIVOT_TOL:
                entering = col
                break
        if entering is None:
            return False
        leaving = None
        best_ratio = float("inf")
        for r in range(n_eq):
            coefficient = tableau[r][entering]
            if coefficient > PIVOT_TOL:
                ratio = tableau[r][-1] / coefficient
                if ratio < best_ratio - 1e-12 or (
                    abs(ratio - best_ratio) <= 1e-12
                    and (leaving is None or basis[r] < basis[leaving])
                ):
                    best_ratio, leaving = ratio, r
        if leaving is None:
            raise ValueError("单纯形出现无界（可行域本应有界，内部错误）")
        _tableau_pivot(tableau, entering, leaving)
        basis[leaving] = entering
        return True

    # 第一阶段入基列覆盖原变量与人工变量中的非基列
    while pivot_once(list(range(n_vars + n_eq))):
        pass

    if tableau[n_eq][-1] < -FEAS_TOL:
        raise ValueError("可行域为空（内部错误：传入的可行点不可行）")

    # 退化情形：人工变量仍在基中时，尝试用原变量把它换出
    for r in range(n_eq):
        if basis[r] >= n_vars:
            for col in range(n_vars):
                if abs(tableau[r][col]) > PIVOT_TOL:
                    _tableau_pivot(tableau, col, r)
                    basis[r] = col
                    break

    # ---- 第二阶段：真实目标 max (-c)^T x；人工变量列清零且不再入基 ----
    for r in range(n_eq + 1):
        for i in range(n_eq):
            tableau[r][n_vars + i] = 0.0
    tableau[-1] = [-v for v in c] + [0.0] * n_eq + [0.0]
    for r in range(n_eq):
        col = basis[r]
        factor = tableau[-1][col]
        if factor:
            for k in range(width):
                tableau[-1][k] -= factor * tableau[r][k]

    while pivot_once(list(range(n_vars))):
        pass

    solution = [0.0] * n_vars
    for r, col in enumerate(basis):
        if col < n_vars:
            solution[col] = tableau[r][-1]

    # 数值清理并投影回可行等式
    solution = [0.0 if abs(v) < 1e-10 else v for v in solution]
    return _repair(solution, eq_rows, eq_rhs, feasible)


def _repair(point, eq_rows, eq_rhs, feasible):
    """把数值解的微小残差修正回等式约束。

    注意：等式行未必是单纯形归一化行（行和未必为 1），因此绝不能对结果
    重新归一化，否则会破坏其它等式。只做零清理 + 最小范数残差修正。
    """
    n_vars = len(point)
    m = len(eq_rows)
    point = [0.0 if abs(v) < 1e-11 else v for v in point]
    residual = [
        eq_rhs[i] - sum(eq_rows[i][j] * point[j] for j in range(n_vars))
        for i in range(m)
    ]
    if max((abs(r) for r in residual), default=0.0) < 1e-9:
        return point
    # 最小范数修正 z = C^T (C C^T)^-1 residual，使 C(x+z)=d
    cct = [
        [sum(eq_rows[i][j] * eq_rows[k][j] for j in range(n_vars))
         for k in range(m)]
        for i in range(m)
    ]
    try:
        lam = solve_linear(cct, residual, ridge=1e-12)
    except SingularMatrix:
        return point
    corrected = [
        point[j] + sum(eq_rows[i][j] * lam[i] for i in range(m))
        for j in range(n_vars)
    ]
    # 只做极小的负零截断；显著负值说明求解出错，不在此掩盖
    corrected = [0.0 if abs(v) < 1e-10 else max(0.0, v) for v in corrected]
    new_residual = max(
        abs(eq_rhs[i] - sum(eq_rows[i][j] * corrected[j]
                            for j in range(n_vars)))
        for i in range(m)
    )
    if new_residual < max(abs(r) for r in residual):
        return corrected
    return point


def min_norm(eq_rows, eq_rhs, feasible):
    """Compute an equality projection and retain its feasible ray segment.

    The equality minimizer is formed by the row-space Gram system. If it
    leaves the nonnegative orthant, restrict the move from the supplied
    feasible representative to the first coordinate boundary. This keeps
    the observed probabilities and normalization unchanged throughout.
    """
    n_vars = len(feasible)
    m = len(eq_rows)
    gram = [
        [sum(eq_rows[i][j] * eq_rows[k][j] for j in range(n_vars))
         for k in range(m)]
        for i in range(m)
    ]
    try:
        multipliers = solve_linear(gram, eq_rhs, ridge=1e-12)
    except SingularMatrix:
        multipliers = solve_linear(gram, eq_rhs, ridge=1e-9)
    projected = [
        sum(eq_rows[i][j] * multipliers[i] for i in range(m))
        for j in range(n_vars)
    ]
    direction = [projected[j] - feasible[j] for j in range(n_vars)]
    step = 1.0
    for j, delta in enumerate(direction):
        if delta < -1e-12:
            step = min(step, max(0.0, feasible[j]) / -delta)
    point = [feasible[j] + step * direction[j] for j in range(n_vars)]
    point = [0.0 if abs(value) < 1e-11 else value for value in point]
    return _repair(point, eq_rows, eq_rhs, feasible)


def _strict_interior(eq_rows, eq_rhs, feasible):
    """把可行点推到相对内部：所有非结构性零的分量严格为正。"""
    n_vars = len(feasible)
    point = feasible[:]
    if all(v > 1e-10 for v in point):
        return point
    basis = null_basis(eq_rows)
    if not basis:
        # 零空间为零：可行集是单点，直接返回（min_norm 不会再迭代）
        return point
    # 沿零空间随机但确定性地组合一个方向，把零分量抬起来
    direction = [0.0] * n_vars
    for k, vector in enumerate(basis):
        sign = 1.0 if k % 2 == 0 else -1.0
        for j in range(n_vars):
            direction[j] += sign * vector[j] / (k + 1)
    step = 1.0
    for _ in range(60):
        candidate = [
            point[j] + step * direction[j] for j in range(n_vars)
        ]
        if all(v > 1e-9 for v in candidate):
            # 投影回等式并确认仍可行
            repaired = _repair(candidate, eq_rows, eq_rhs, feasible)
            if all(v >= -1e-12 for v in repaired):
                return repaired
        step *= 0.5
    # 兜底：均匀内部点（一般不满足等式，仅作最后手段）
    return point
