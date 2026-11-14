"""仿射多面体上的线性与二次规划（全部手写，不调用求解器）。

MLE 集合是仿射切片与单纯形的交：

    P = { w : C w = d, w >= 0 }

其中 ``C`` 由一个归一化等式与若干 ``M w = q*`` 等式堆叠而成。
本模块提供：

* :func:`null_basis`   —— 行约化求零空间基；
* :func:`linear_program` —— 两阶段单纯形，返回顶点最优解；
* :func:`min_norm`    —— 主动集法解 ``min sum w_j^2`` s.t. ``C w = d, w >= 0``
  （严格凸 QP，解唯一），带单纯形 LP 最优性证书。
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


# 主动集 QP 的数值容差（远小于输出容差 1e-6）
QP_STEP_TOL = 1e-12   # 面内位移视为零
QP_BLOCK_TOL = 1e-15  # 方向分量视为下降（可能触界）
QP_CERT_TOL = 1e-9    # 最优性证书（相对）容差


def _independent_rows(matrix, rhs):
    """对增广矩阵 ``[matrix | rhs]`` 做 Gauss-Jordan 行约化。

    返回主元行对应的 ``(rows, values)``：系数行满秩，且在系统一致时
    与原系统等价（非主元行被消为零行而丢弃）。容差与
    :func:`solver._rref` 一致。
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0
    aug = [list(matrix[r]) + [rhs[r]] for r in range(n_rows)]
    scale = max(
        (abs(aug[r][c]) for r in range(n_rows) for c in range(n_cols)),
        default=1.0,
    )
    tol = 1e-11 * max(1.0, scale)
    pivot_rows = []
    row_idx = 0
    for col in range(n_cols):
        pivot = max(
            range(row_idx, n_rows),
            key=lambda r: abs(aug[r][col]),
            default=None,
        )
        if pivot is None or abs(aug[pivot][col]) <= tol:
            continue
        aug[row_idx], aug[pivot] = aug[pivot], aug[row_idx]
        pivot_value = aug[row_idx][col]
        aug[row_idx] = [v / pivot_value for v in aug[row_idx]]
        for r in range(n_rows):
            if r == row_idx:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [
                    aug[r][c] - factor * aug[row_idx][c]
                    for c in range(n_cols + 1)
                ]
        pivot_rows.append(row_idx)
        row_idx += 1
        if row_idx == n_rows:
            break
    rows = [aug[r][:n_cols] for r in pivot_rows]
    values = [aug[r][n_cols] for r in pivot_rows]
    return rows, values


def _face_min_norm(eq_rows, eq_rhs, zero, n_vars):
    """求仿射面 ``{x : C x = d, x_j = 0 (j in zero)}`` 的最小范数点。

    调用方保证系统一致（当前可行点的支撑含于自由坐标）。先把
    ``C_F x = d`` 行约化为满行秩系统 ``R x = b``，则最小范数解为
    ``x = R^T (R R^T)^{-1} b``（Gram 系统对称正定，精确求解）。
    """
    free = [j for j in range(n_vars) if j not in zero]
    if not free:
        raise RuntimeError("内部错误：主动集 QP 的自由坐标为空")
    sub = [[row[j] for j in free] for row in eq_rows]
    rows, values = _independent_rows(sub, eq_rhs)
    m = len(rows)
    gram = [
        [sum(rows[i][k] * rows[j][k] for k in range(len(free)))
         for j in range(m)]
        for i in range(m)
    ]
    try:
        multipliers = solve_linear(gram, values)
    except SingularMatrix:
        multipliers = solve_linear(gram, values, ridge=1e-12)
    point = [0.0] * n_vars
    for k, j in enumerate(free):
        point[j] = sum(rows[i][k] * multipliers[i] for i in range(m))
    return point


def min_norm(eq_rows, eq_rhs, feasible):
    """在 ``P = {w : C w = d, w >= 0}`` 上求 ``sum w_j^2`` 的唯一最小点。

    手写原始主动集法，不在任何一步用"可行但非最优"的点充数：

    1. **面内阶段**：在当前固定零集 ``Z`` 的仿射面上求最小范数点
       （:func:`_face_min_norm`）。若该点越出非负象限，从当前可行点
       沿该方向做比率步（ratio test），把触零坐标全部并入 ``Z`` 后
       重解，直到当前点就是面内最优点。``Z`` 只增不减，故面内阶段
       有限步结束，且目标值严格下降。
    2. **最优性证书**：``w`` 是凸集 ``P`` 上的最小范数点，当且仅当
       对一切 ``v in P`` 有 ``<w, v> >= ||w||^2``（投影的变分刻画）。
       用手写两阶段单纯形解 ``min <w, v>`` 验证；证书通过即当前点是
       全 MLE 集合上的唯一最小范数解。
    3. **证书修正**：若证书失败，顶点 ``v`` 给出严格下降方向，沿线段
       ``[w, v]`` 做精确线搜索（线段仍在 ``P`` 内），步长严格为正，
       目标严格下降且零集严格收缩，随后回到面内阶段。

    目标严格凸保证解唯一；每次外层迭代要么结束于某张面的最小范数点
    （由零集唯一确定），要么严格降低 ``||w||``，同一零集不会重复
    出现，算法有限终止。返回前再做一次等式残差修正。
    """
    n_vars = len(feasible)

    def clean(point):
        # 只截断数值噪声产生的微小分量，绝不重新归一化
        return [v if v > 1e-12 else 0.0 for v in point]

    # 起点：调用方给出的可行 MLE 权重（先投影回等式并清理噪声）
    w = clean(_repair(list(feasible), eq_rows, eq_rhs, feasible))
    zero = {j for j in range(n_vars) if w[j] <= 0.0}

    max_outer = 200 * (n_vars + 2)
    for _ in range(max_outer):
        # ---- 面内阶段：求当前零集所定仿射面的最小范数点 ----
        for _ in range(n_vars + 2):
            y = _face_min_norm(eq_rows, eq_rhs, zero, n_vars)
            direction = [y[j] - w[j] for j in range(n_vars)]
            if max(abs(v) for v in direction) <= QP_STEP_TOL:
                w = clean(y)
                break
            # 比率步：在不越出非负象限的前提下尽量走向 y
            alpha = 1.0
            for j in range(n_vars):
                if j not in zero and direction[j] < -QP_BLOCK_TOL:
                    alpha = min(alpha, w[j] / -direction[j])
            w = clean([w[j] + alpha * direction[j] for j in range(n_vars)])
            if alpha >= 1.0:
                break
            # 触零坐标（含并列）全部并入固定零集
            zero.update(j for j in range(n_vars) if w[j] <= 0.0)
        else:
            raise RuntimeError("内部错误：主动集 QP 面内迭代未收敛")

        # ---- 最优性证书：min <w, v> >= ||w||^2 ? ----
        vertex = linear_program(list(w), eq_rows, eq_rhs, w)
        norm_sq = sum(v * v for v in w)
        violation = norm_sq - sum(w[j] * vertex[j] for j in range(n_vars))
        if violation <= QP_CERT_TOL * max(1.0, norm_sq):
            return _repair(w, eq_rows, eq_rhs, feasible)

        # ---- 证书修正：沿 [w, vertex] 精确线搜索，严格减小 ||w|| ----
        diff = [w[j] - vertex[j] for j in range(n_vars)]
        denom = sum(v * v for v in diff)
        if denom <= 1e-24:
            # 顶点与当前点数值上重合，无法再改进
            return _repair(w, eq_rows, eq_rhs, feasible)
        step = sum(w[j] * diff[j] for j in range(n_vars)) / denom
        step = min(1.0, max(0.0, step))
        w = clean([w[j] - step * diff[j] for j in range(n_vars)])
        zero = {j for j in range(n_vars) if w[j] <= 0.0}

    raise RuntimeError("内部错误：主动集 QP 超过迭代上限")
