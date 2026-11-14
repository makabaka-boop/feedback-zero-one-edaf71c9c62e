"""代表解（最小二乘范数 QP）的确定性回归与独立精确枚举核对。

背景：代表解必须是**全部**非负、归一化且达到同一最大似然的共享权重中
``sum w_j^2`` 最小的唯一解。拟合概率与似然相同只能说明它落在 MLE 集合
内，不能说明它最优——本文件的核对均不依赖"似然一致"这条弱判据：

1. 手算小例：题目给出的两个单实验四类案例，代表、拟合概率、各类范围
   端点都有闭式精确解；
2. 独立精确枚举：用 :mod:`fractions` 精确分数对 MLE 多面体做支撑面
   枚举（严格凸目标的唯一最优点必为自身支撑面的仿射最小范数点），
   与求解器结果逐一比对——覆盖有零分量的最优代表、冗余观测约束、
   相同响应列、唯一可辨识解；
3. 列重排不变性：classes 与 A 的列同步重排后，代表与范围端点值只作
   相应重排（端点见证仍按原规则任取，只核对合法性）；
4. 回归：相同两列代表均分；单位响应矩阵下权重由计数唯一确定。

运行：python3 -I tests/test_representative_qp.py
"""

import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from mixture_est.em import compute_q, log_likelihood
from mixture_est.solver import _prepare, estimate
from mixture_est.validation import validate

TOL = 1e-6
L_TOL = 1e-8
_failures = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        _failures.append((name, detail))
        print("  FAIL  %s  %s" % (name, detail))


def approx(a, b, tol=TOL):
    return abs(a - b) <= tol


def vec_approx(a, b, tol=TOL):
    return len(a) == len(b) and all(approx(x, y, tol) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 独立精确枚举（Fraction 精确算术，与求解器浮点实现完全独立）
# ---------------------------------------------------------------------------

def exact_min_norm(rows, rhs):
    """``{x : C x = d, x >= 0}`` 上 ``sum x_j^2`` 的精确最小点。

    严格凸目标的唯一最优点必为自身支撑面的仿射最小范数点；枚举全部
    支撑面，取可行的仿射最小范数点中范数最小者即全局最优。
    ``rows``/``rhs`` 均为 :class:`Fraction`。
    """
    n = len(rows[0])
    best_x, best_norm = None, None
    for mask in range(1, 1 << n):
        support = [j for j in range(n) if mask >> j & 1]
        sub = [[row[j] for j in support] for row in rows]
        m = len(sub)
        # 精确解 C_S C_S^T lam = d（取任一特解；x = C_S^T lam 唯一）
        gram = [[sum(sub[i][k] * sub[j][k] for k in range(len(support)))
                 for j in range(m)] for i in range(m)]
        aug = [gram[i] + [rhs[i]] for i in range(m)]
        piv_cols, r = [], 0
        for col in range(m):
            piv = next((rr for rr in range(r, m) if aug[rr][col] != 0), None)
            if piv is None:
                continue
            aug[r], aug[piv] = aug[piv], aug[r]
            pivot_value = aug[r][col]
            aug[r] = [v / pivot_value for v in aug[r]]
            for rr in range(m):
                if rr != r and aug[rr][col] != 0:
                    factor = aug[rr][col]
                    aug[rr] = [aug[rr][c] - factor * aug[r][c]
                               for c in range(m + 1)]
            piv_cols.append(col)
            r += 1
        if any(all(aug[rr][c] == 0 for c in range(m)) and aug[rr][m] != 0
               for rr in range(m)):
            continue  # 该支撑面不可行
        lam = [Fraction(0)] * m
        for rr, col in enumerate(piv_cols):
            lam[col] = aug[rr][m]
        x = [sum(sub[i][k] * lam[i] for i in range(m))
             for k in range(len(support))]
        if any(v <= 0 for v in x):
            continue  # 不在该面的相对内部
        full = [Fraction(0)] * n
        for k, j in enumerate(support):
            full[j] = x[k]
        norm = sum(v * v for v in full)
        if best_norm is None or norm < best_norm:
            best_norm, best_x = norm, full
    return best_x, best_norm


def exact_vertices(rows, rhs):
    """枚举 ``{C x = d, x >= 0}`` 的全部顶点（基本可行解，精确分数）。"""
    from itertools import combinations
    n = len(rows[0])
    m = len(rows)
    vertices = []
    for basis in combinations(range(n), m):
        sub = [[row[j] for j in basis] for row in rows]
        # 精确解 C_B x = d（方阵；奇异则跳过）
        aug = [sub[i] + [rhs[i]] for i in range(m)]
        piv_cols, r = [], 0
        for col in range(m):
            piv = next((rr for rr in range(r, m) if aug[rr][col] != 0), None)
            if piv is None:
                continue
            aug[r], aug[piv] = aug[piv], aug[r]
            pivot_value = aug[r][col]
            aug[r] = [v / pivot_value for v in aug[r]]
            for rr in range(m):
                if rr != r and aug[rr][col] != 0:
                    factor = aug[rr][col]
                    aug[rr] = [aug[rr][c] - factor * aug[r][c]
                               for c in range(m + 1)]
            piv_cols.append(col)
            r += 1
        if len(piv_cols) < m:
            continue  # 奇异基
        x = [Fraction(0)] * n
        for rr, col in enumerate(piv_cols):
            x[basis[col]] = aug[rr][m]
        if any(v < 0 for v in x):
            continue
        if not any(all(v == u for v, u in zip(x, old)) for old in vertices):
            vertices.append(x)
    return vertices


def check_against_exact_oracle(result, rows, rhs, label, tol=TOL):
    """用精确枚举核对代表解的可行性与最优性、范围端点值。"""
    n = len(rows[0])
    rep = result["representative"]

    # 可行性：非负、归一化、满足全部 MLE 等式（用求解器自己的浮点等式）
    check("%s：代表非负" % label, all(v >= -TOL for v in rep))
    check("%s：代表归一化" % label, approx(sum(rep), 1.0, 1e-9))
    for i, row in enumerate(rows):
        residual = abs(sum(float(row[j]) * rep[j] for j in range(n))
                       - float(rhs[i]))
        check("%s：代表满足等式 %d" % (label, i), residual <= 1e-6,
              "残差 %.3g" % residual)

    # 最优性：与精确最小范数点逐分量一致
    best_x, best_norm = exact_min_norm(rows, rhs)
    check("%s：精确枚举存在最优点" % label, best_x is not None)
    if best_x is None:
        return
    check("%s：代表=精确最小范数点" % label,
          vec_approx(rep, [float(v) for v in best_x], tol),
          "rep=%s exact=%s" % (rep, [float(v) for v in best_x]))
    rep_norm = sum(v * v for v in rep)
    check("%s：代表范数=精确最小范数" % label,
          approx(rep_norm, float(best_norm), tol),
          "%.12g vs %.12g" % (rep_norm, float(best_norm)))

    # 范围端点：与顶点枚举的逐坐标最小/最大一致
    vertices = exact_vertices(rows, rhs)
    if vertices:
        for j in range(n):
            lo = min(v[j] for v in vertices)
            hi = max(v[j] for v in vertices)
            check("%s：类别 %d 范围端点精确" % (label, j),
                  approx(result["ranges"][j]["minimum"], float(lo), tol)
                  and approx(result["ranges"][j]["maximum"], float(hi), tol),
                  "[%s, %s] vs exact [%s, %s]"
                  % (result["ranges"][j]["minimum"],
                     result["ranges"][j]["maximum"], float(lo), float(hi)))


# ---------------------------------------------------------------------------
# 1. 手算小例（题目给出的两个案例）
# ---------------------------------------------------------------------------

def test_spec_case_zero_component_rep():
    print("\n[1] 手算例：n=[1,4]，最优代表含零分量 [8/15,1/3,2/15,0]")
    data = {
        "classes": ["c1", "c2", "c3", "c4"],
        "experiments": [
            {"id": "e1",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [1, 4]}
        ],
    }
    validated = validate(data)
    result = estimate(validated)
    check("status=ok", result["status"] == "ok")
    check("拟合概率 [1/5,4/5]",
          vec_approx(result["fitted_probabilities"][0]["q"], [0.2, 0.8]))
    expected = [8 / 15, 1 / 3, 2 / 15, 0.0]
    check("代表 [8/15,1/3,2/15,0]",
          vec_approx(result["representative"], expected),
          "got %s" % result["representative"])
    check("代表范数 93/225",
          approx(sum(v * v for v in result["representative"]), 93 / 225))
    expected_l = (math.log(0.2) + 4 * math.log(0.8)) / 5
    check("归一化似然", approx(result["log_likelihood"], expected_l, L_TOL))
    # 手算范围端点：w1∈[2/5,4/5], w2∈[0,3/5], w3∈[0,3/10], w4∈[0,1/5]
    endpoints = [(0.4, 0.8), (0.0, 0.6), (0.0, 0.3), (0.0, 0.2)]
    for j, (lo, hi) in enumerate(endpoints):
        check("类别 %d 范围 [%g,%g]" % (j + 1, lo, hi),
              approx(result["ranges"][j]["minimum"], lo)
              and approx(result["ranges"][j]["maximum"], hi))
    # 独立精确枚举（多面体用精确 q*=[1/5,4/5]）
    rows = [[Fraction(1)] * 4,
            [Fraction(0), Fraction(1, 3), Fraction(2, 3), Fraction(1)],
            [Fraction(1), Fraction(2, 3), Fraction(1, 3), Fraction(0)]]
    rhs = [Fraction(1), Fraction(1, 5), Fraction(4, 5)]
    check_against_exact_oracle(result, rows, rhs, "n=[1,4]")


def test_spec_case_boundary_rep():
    print("\n[2] 手算例：n=[1,5]，最优代表 [7/12,1/3,1/12,0]（非 [1/2,1/2,0,0]）")
    data = {
        "classes": ["c1", "c2", "c3", "c4"],
        "experiments": [
            {"id": "e1",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [1, 5]}
        ],
    }
    validated = validate(data)
    result = estimate(validated)
    q = result["fitted_probabilities"][0]["q"]
    check("拟合概率 [1/6,5/6]", vec_approx(q, [1 / 6, 5 / 6]))
    expected = [7 / 12, 1 / 3, 1 / 12, 0.0]
    check("代表 [7/12,1/3,1/12,0]",
          vec_approx(result["representative"], expected),
          "got %s" % result["representative"])
    check("代表范数 11/24",
          approx(sum(v * v for v in result["representative"]), 11 / 24))
    # 错误代表 [1/2,1/2,0,0] 也在 MLE 集合内（似然相同）但范数 1/2 更大：
    # 明确核对代表范数严格小于该可行非最优点
    check("代表范数严格小于 1/2",
          sum(v * v for v in result["representative"]) < 0.5 - 1e-9)
    endpoints = [(0.5, 5 / 6), (0.0, 0.5), (0.0, 0.25), (0.0, 1 / 6)]
    for j, (lo, hi) in enumerate(endpoints):
        check("类别 %d 范围端点" % (j + 1),
              approx(result["ranges"][j]["minimum"], lo)
              and approx(result["ranges"][j]["maximum"], hi))
    rows = [[Fraction(1)] * 4,
            [Fraction(0), Fraction(1, 3), Fraction(2, 3), Fraction(1)],
            [Fraction(1), Fraction(2, 3), Fraction(1, 3), Fraction(0)]]
    rhs = [Fraction(1), Fraction(1, 6), Fraction(5, 6)]
    check_against_exact_oracle(result, rows, rhs, "n=[1,5]")


# ---------------------------------------------------------------------------
# 2. 冗余观测约束
# ---------------------------------------------------------------------------

def test_redundant_constraints():
    print("\n[3] 冗余观测约束：重复实验/重复行不改变代表与范围")
    base = {
        "classes": ["c1", "c2", "c3", "c4"],
        "experiments": [
            {"id": "e1",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [1, 4]}
        ],
    }
    # 同一实验重复三次（计数同比例放大），MLE 集合不变
    redundant = {
        "classes": ["c1", "c2", "c3", "c4"],
        "experiments": [
            {"id": "e1",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [1, 4]},
            {"id": "e2",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [2, 8]},
            {"id": "e3",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [3, 12]},
        ],
    }
    # 观测行拆分：r -> r/2 + r/2（列和仍为 1），计数相应拆分；
    # 似然只差常数 -(n1+n2)ln2/N，MLE 集合与代表不变
    split_rows = {
        "classes": ["c1", "c2", "c3", "c4"],
        "experiments": [
            {"id": "e1",
             "A": [["0/1", "1/6", "1/3", "1/2"],
                   ["0/1", "1/6", "1/3", "1/2"],
                   ["1/2", "1/3", "1/6", "0/1"],
                   ["1/2", "1/3", "1/6", "0/1"]],
             "n": [1, 0, 3, 1]}
        ],
    }
    ref = estimate(validate(base))
    for label, variant in [("重复实验", redundant), ("拆分观测行", split_rows)]:
        result = estimate(validate(variant))
        check("%s：代表与基准一致" % label,
              vec_approx(result["representative"], ref["representative"]),
              "%s vs %s" % (result["representative"], ref["representative"]))
        for j in range(4):
            check("%s：类别 %d 端点一致" % (label, j),
                  approx(result["ranges"][j]["minimum"],
                         ref["ranges"][j]["minimum"])
                  and approx(result["ranges"][j]["maximum"],
                             ref["ranges"][j]["maximum"]))
    check("重复实验：似然一致",
          approx(estimate(validate(redundant))["log_likelihood"],
                 ref["log_likelihood"], L_TOL))
    # 拆分行的拟合概率是基准对应行的一半
    split_result = estimate(validate(split_rows))
    base_q = ref["fitted_probabilities"][0]["q"]
    split_q = split_result["fitted_probabilities"][0]["q"]
    check("拆分行：拟合概率减半",
          vec_approx(split_q, [base_q[0] / 2, base_q[0] / 2,
                               base_q[1] / 2, base_q[1] / 2]))


# ---------------------------------------------------------------------------
# 3. 相同响应列（回归：代表均分）
# ---------------------------------------------------------------------------

def test_identical_columns_split_evenly():
    print("\n[4] 相同响应列：代表均分（回归）+ 精确枚举核对")
    data = {
        "classes": ["x", "y"],
        "experiments": [
            {"id": "e",
             "A": [["1/3", "1/3"], ["2/3", "2/3"]],
             "n": [10, 20]},
        ],
    }
    result = estimate(validate(data))
    check("两列代表均分", vec_approx(result["representative"], [0.5, 0.5]))
    check("代表范数 1/2",
          approx(sum(v * v for v in result["representative"]), 0.5, 1e-9))

    data3 = {
        "classes": ["a", "b", "c"],
        "experiments": [
            {"id": "e",
             "A": [["1/3", "1/3", "1/3"], ["2/3", "2/3", "2/3"]],
             "n": [10, 20]},
        ],
    }
    result3 = estimate(validate(data3))
    check("三列代表均分",
          vec_approx(result3["representative"], [1 / 3, 1 / 3, 1 / 3]))
    rows = [[Fraction(1)] * 3,
            [Fraction(1, 3)] * 3, [Fraction(2, 3)] * 3]
    rhs = [Fraction(1), Fraction(1, 3), Fraction(2, 3)]
    check_against_exact_oracle(result3, rows, rhs, "三相同列")


# ---------------------------------------------------------------------------
# 4. 唯一可辨识解（回归：单位响应矩阵权重由计数唯一确定）
# ---------------------------------------------------------------------------

def test_identity_matrix_unique_weights():
    print("\n[5] 单位响应矩阵：权重由计数唯一确定，范围退化为点")
    data = {
        "classes": ["a", "b", "c"],
        "experiments": [
            {"id": "e",
             "A": [["1/1", "0/1", "0/1"],
                   ["0/1", "1/1", "0/1"],
                   ["0/1", "0/1", "1/1"]],
             "n": [2, 3, 5]},
        ],
    }
    result = estimate(validate(data))
    check("代表=计数比例",
          vec_approx(result["representative"], [0.2, 0.3, 0.5]))
    for j, p in enumerate([0.2, 0.3, 0.5]):
        check("类别 %d 范围退化为 %g" % (j, p),
              approx(result["ranges"][j]["minimum"], p)
              and approx(result["ranges"][j]["maximum"], p))
        check("类别 %d 见证=代表" % j,
              vec_approx(result["ranges"][j]["min_witness"],
                         result["representative"])
              and vec_approx(result["ranges"][j]["max_witness"],
                             result["representative"]))
    rows = [[Fraction(1)] * 3,
            [Fraction(1), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(1), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(1)]]
    rhs = [Fraction(1), Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)]
    check_against_exact_oracle(result, rows, rhs, "单位矩阵")


# ---------------------------------------------------------------------------
# 5. 列同步重排不变性
# ---------------------------------------------------------------------------

def test_column_permutation_invariance():
    print("\n[6] classes 与 A 的列同步重排：代表与范围端点只作相应重排")
    base = {
        "classes": ["c1", "c2", "c3", "c4"],
        "experiments": [
            {"id": "e1",
             "A": [["0/1", "1/3", "2/3", "1/1"],
                   ["1/1", "2/3", "1/3", "0/1"]],
             "n": [1, 4]},
            {"id": "e2",
             "A": [["1/2", "1/4", "1/8", "0/1"],
                   ["1/2", "3/4", "7/8", "1/1"]],
             "n": [3, 2]},
        ],
    }
    perm = [2, 0, 3, 1]  # new_classes[i] = base.classes[perm[i]]
    base_result = estimate(validate(base))

    permuted_classes = [base["classes"][p] for p in perm]
    permuted_experiments = []
    for experiment in base["experiments"]:
        permuted_experiments.append({
            "id": experiment["id"],
            "A": [[row[p] for p in perm] for row in experiment["A"]],
            "n": list(experiment["n"]),
        })
    permuted = {"classes": permuted_classes,
                "experiments": permuted_experiments}
    perm_result = estimate(validate(permuted))

    # 代表：重排后的结果应等于基准结果的重排
    expected_rep = [base_result["representative"][p] for p in perm]
    check("代表按列重排",
          vec_approx(perm_result["representative"], expected_rep),
          "%s vs %s" % (perm_result["representative"], expected_rep))
    check("似然不变",
          approx(perm_result["log_likelihood"],
                 base_result["log_likelihood"], L_TOL))

    # 范围端点值按列重排；端点见证只核对合法性（任取规则不变）
    for i, p in enumerate(perm):
        lo = base_result["ranges"][p]["minimum"]
        hi = base_result["ranges"][p]["maximum"]
        check("类别 %s 端点重排" % permuted_classes[i],
              approx(perm_result["ranges"][i]["minimum"], lo)
              and approx(perm_result["ranges"][i]["maximum"], hi))
        for key in ("min_witness", "max_witness"):
            witness = perm_result["ranges"][i][key]
            check("见证归一化/非负",
                  approx(sum(witness), 1.0, 1e-9)
                  and all(v >= -1e-9 for v in witness))
        check("min 见证达到最小端点",
              approx(perm_result["ranges"][i]["min_witness"][i],
                     perm_result["ranges"][i]["minimum"]))
        check("max 见证达到最大端点",
              approx(perm_result["ranges"][i]["max_witness"][i],
                     perm_result["ranges"][i]["maximum"]))

    # 见证似然与基准一致（仍在 MLE 集合上）
    validated = validate(permuted)
    (_, _, flat, offsets, rows, counts, total_n) = _prepare(validated)
    for r in perm_result["ranges"]:
        for witness in (r["min_witness"], r["max_witness"]):
            q = compute_q(flat, offsets, rows, witness)
            check("见证似然一致",
                  approx(log_likelihood(counts, q, total_n),
                           perm_result["log_likelihood"], L_TOL))


# ---------------------------------------------------------------------------
# 6. 直接对 min_norm 的精确 QP 测试（不经过 EM，排除 q* 浮点精度影响）
# ---------------------------------------------------------------------------

def test_min_norm_direct_exact():
    print("\n[7] min_norm 直接精确测试：给定多面体求最小二乘范数点")
    from mixture_est.polytope import min_norm

    # (说明, C 行, d, 初始可行点, 精确最优点)
    cases = [
        # 题目案例的多面体；初始点取旧实现的错误代表 [0.55,0.3,0.15,0]，
        # 确保不会把"可行但非最优"的点当作答案
        ("题目多面体 q=[1/5,4/5]",
         [[1, 1, 1, 1], [0, "1/3", "2/3", 1], [1, "2/3", "1/3", 0]],
         [1, "1/5", "4/5"],
         ["11/20", "3/10", "3/20", 0],
         ["8/15", "1/3", "2/15", 0]),
        # 同一多面体，初始可行点的零集与最优点不同（强制走证书修正路径）
        ("题目多面体 错误零集起点",
         [[1, 1, 1, 1], [0, "1/3", "2/3", 1], [1, "2/3", "1/3", 0]],
         [1, "1/5", "4/5"],
         ["2/5", "3/5", 0, 0],
         ["8/15", "1/3", "2/15", 0]),
        # 相同响应列：两列完全不可分，最小范数点均分
        ("相同两列均分",
         [[1, 1, 1], ["1/2", "1/2", "1/4"]],
         [1, "2/5"],
         ["3/5", 0, "2/5"],
         ["3/10", "3/10", "2/5"]),
        # 唯一可行点（点状多面体）：代表只能是它
        ("唯一可行点",
         [[1, 1], [1, 2]],
         [1, "3/2"],
         ["1/2", "1/2"],
         ["1/2", "1/2"]),
        # 边界起点 + 内部最优点（三等分）
        ("边界起点内部最优",
         [[1, 1, 1], [2, 1, 0]],
         [1, 1],
         [0, 1, 0],
         ["1/3", "1/3", "1/3"]),
        # 冗余等式行（第二行是第一行的倍数）：最优点不变
        ("冗余等式行",
         [[1, 1, 1], ["1/2", "1/2", "1/2"], ["1/5", "1/2", "4/5"]],
         [1, "1/2", "1/2"],
         [0, 1, 0],
         ["1/3", "1/3", "1/3"]),
        # 最优含零分量，且起点的零集错误（需经证书修正释放坐标）
        ("零分量最优+释放",
         [[1, 1, 1], [0, "1/2", 1]],
         [1, "1/6"],
         ["5/6", 0, "1/6"],
         ["2/3", "1/3", 0]),
    ]
    for label, c_rows, d_vals, start, expected in cases:
        rows_f = [[Fraction(str(v)) for v in row] for row in c_rows]
        rhs_f = [Fraction(str(v)) for v in d_vals]
        start_f = [Fraction(str(v)) for v in start]
        # 确认初始点可行（精确）
        for i, row in enumerate(rows_f):
            assert sum(row[j] * start_f[j] for j in range(len(start_f))) == rhs_f[i], \
                "用例 %s 的初始点不可行" % label
        eq_rows = [[float(v) for v in row] for row in rows_f]
        eq_rhs = [float(v) for v in rhs_f]
        feasible = [float(v) for v in start_f]
        got = min_norm(eq_rows, eq_rhs, feasible)
        exp = [float(Fraction(str(v))) for v in expected]
        check("%s：min_norm=精确点" % label, vec_approx(got, exp, 1e-9),
              "got %s want %s" % (got, exp))
        # 与独立精确枚举一致
        best_x, best_norm = exact_min_norm(rows_f, rhs_f)
        check("%s：与精确枚举一致" % label,
              best_x is not None
              and vec_approx(got, [float(v) for v in best_x], 1e-9))
        # 可行性残差
        for i, row in enumerate(eq_rows):
            residual = abs(sum(row[j] * got[j] for j in range(len(got)))
                           - eq_rhs[i])
            check("%s：等式 %d 残差" % (label, i), residual <= 1e-9)


# ---------------------------------------------------------------------------
# 7. 随机小例的独立精确枚举（确定性种子）
# ---------------------------------------------------------------------------
def test_random_exact_crosscheck():
    print("\n[8] 随机单实验两行小例：精确枚举核对代表可行性/最优性")
    import random
    rng = random.Random(20261114)
    checked = 0
    for _ in range(60):
        k = rng.randint(2, 4)
        row0 = [Fraction(rng.randint(0, 12), 12) for _ in range(k)]
        row1 = [Fraction(1) - v for v in row0]
        n1 = rng.randint(1, 20)
        n2 = rng.randint(1, 20)
        q1 = Fraction(n1, n1 + n2)
        # 仅当 q*=(n1/N,n2/N) 严格可达时，精确 MLE 集合才是
        # {sum w=1, row0.w=q1}（否则 q* 落在边界上，不属于本族）
        if not (min(row0) < q1 < max(row0)):
            continue
        data = {
            "classes": ["c%d" % j for j in range(k)],
            "experiments": [{
                "id": "e",
                "A": [["%d/%d" % (v.numerator, v.denominator) for v in row0],
                      ["%d/%d" % (v.numerator, v.denominator) for v in row1]],
                "n": [n1, n2],
            }],
        }
        result = estimate(validate(data))
        # EM 的 q* 精度（独立于 QP）：拟合概率应在 1e-6 内等于精确 q*
        check("随机例 k=%d n=[%d,%d]：拟合概率精确" % (k, n1, n2),
              approx(result["fitted_probabilities"][0]["q"][0], float(q1)))
        # 代表/端点与精确枚举的比对容差放宽到 2e-5：残余差异来自 EM 的
        # q* 浮点精度（影响多面体本身，范围端点同样受影响），不是 QP 误差；
        # QP 在给定多面体上的精确性由 [8] 的直接精确测试保证
        rows = [[Fraction(1)] * k, list(row0)]
        rhs = [Fraction(1), q1]
        check_against_exact_oracle(result, rows, rhs,
                                   "随机例 k=%d n=[%d,%d]" % (k, n1, n2),
                                   tol=2e-5)
        checked += 1
    print("  （共精确枚举核对 %d 组）" % checked)
    check("至少核对 20 组", checked >= 20, "checked=%d" % checked)


def main():
    tests = [
        test_spec_case_zero_component_rep,
        test_spec_case_boundary_rep,
        test_redundant_constraints,
        test_identical_columns_split_evenly,
        test_identity_matrix_unique_weights,
        test_column_permutation_invariance,
        test_min_norm_direct_exact,
        test_random_exact_crosscheck,
    ]
    for test in tests:
        test()
    print("\n" + "=" * 60)
    if _failures:
        print("失败 %d 项：" % len(_failures))
        for name, detail in _failures:
            print("  - %s  %s" % (name, detail))
        return 1
    print("代表解 QP 回归全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
