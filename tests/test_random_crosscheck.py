"""随机交叉验证：用独立的坐标上升 + 密集网格核对 EM/MLE、范围、代表解。

运行：python3 -I tests/test_random_crosscheck.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from mixture_est.em import compute_q, log_likelihood
from mixture_est.errors import ImpossibleEstimation
from mixture_est.solver import _prepare, estimate
from mixture_est.validation import validate


def random_column_stochastic(rng, rows, cols):
    """随机列随机矩阵，用分数保持列和精确为 1。"""
    matrix = [[None] * cols for _ in range(rows)]
    for j in range(cols):
        cuts = sorted(rng.randrange(1, 1000) for _ in range(rows - 1))
        bounds = [0] + cuts + [1000]
        for i in range(rows):
            matrix[i][j] = "%d/1000" % (bounds[i + 1] - bounds[i])
    return matrix


def coordinate_ascent(flat, offsets, rows, counts, N, k, rng, rounds=4000):
    """独立参考实现：在单纯形上做块坐标上升求 MLE。"""
    w = [1.0 / k] * k
    q = compute_q(flat, offsets, rows, w)
    best = log_likelihood(counts, q, N)
    for _ in range(rounds):
        i, j = rng.sample(range(k), 2)

        def value(wi):
            trial = w[:]
            mass = w[i] + w[j]
            trial[i] = wi
            trial[j] = mass - wi
            if wi < 0 or trial[j] < 0:
                return -math.inf
            qq = compute_q(flat, offsets, rows, trial)
            return log_likelihood(counts, qq, N)

        lo, hi = 0.0, w[i] + w[j]
        for _ in range(40):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if value(m1) < value(m2):
                lo = m1
            else:
                hi = m2
        wi = (lo + hi) / 2
        mass = w[i] + w[j]
        w[i] = wi
        w[j] = mass - wi
        q = compute_q(flat, offsets, rows, w)
        best = log_likelihood(counts, q, N)
    return w, best


def run(trials=40, seed=20261113):
    rng = random.Random(seed)
    failures = 0

    for trial in range(trials):
        k = rng.randint(2, 5)
        classes = ["c%d" % j for j in range(k)]
        experiments = []
        for e in range(rng.randint(1, 3)):
            rows_count = rng.randint(2, 4)
            A = random_column_stochastic(rng, rows_count, k)
            n = [rng.randint(0, 25) for _ in range(rows_count)]
            if sum(n) == 0:
                n[0] = 1
            experiments.append({"id": "e%d" % e, "A": A, "n": n})

        data = {"classes": classes, "experiments": experiments}
        try:
            result = estimate(validate(data))
        except ImpossibleEstimation:
            continue  # 合法结果

        _, kk, flat, offsets, rows, counts, N = _prepare(validate(data))

        # 1) 独立坐标上升的似然不得明显超过 EM 的 MLE
        _, ref_l = coordinate_ascent(flat, offsets, rows, counts, N, kk, rng)
        if ref_l > result["log_likelihood"] + 1e-7:
            print("试验 %d FAIL: 参考似然 %.8f > MLE %.8f"
                  % (trial, ref_l, result["log_likelihood"]))
            failures += 1
            continue

        # 2) 所有见证的似然必须等于 MLE（LP/QP 可行性）
        points = [result["representative"]]
        for r in result["ranges"]:
            points += [r["min_witness"], r["max_witness"]]
        for p in points:
            q = compute_q(flat, offsets, rows, p)
            ll = log_likelihood(counts, q, N)
            if abs(ll - result["log_likelihood"]) > 1e-8:
                print("试验 %d FAIL: 见证似然差 %.2e"
                      % (trial, ll - result["log_likelihood"]))
                failures += 1

        # 3) 代表解二乘范数不大于任何见证
        rep_norm = sum(v * v for v in result["representative"])
        for p in points:
            if rep_norm > sum(v * v for v in p) + 1e-9:
                print("试验 %d FAIL: 代表解非最小范数" % trial)
                failures += 1
                break

        # 4) 范围端点次序，且所有见证落在范围内
        for j, r in enumerate(result["ranges"]):
            if r["minimum"] > r["maximum"] + 1e-9:
                print("试验 %d FAIL: 范围次序" % trial)
                failures += 1
            for p in points:
                if p[j] < r["minimum"] - 1e-6 or p[j] > r["maximum"] + 1e-6:
                    print("试验 %d FAIL: 见证分量 %.6f 超出 [%g,%g]"
                          % (trial, p[j], r["minimum"], r["maximum"]))
                    failures += 1

        # 5) k<=3 时密集网格核对最小范数代表
        if kk <= 3:
            steps = 120
            best_norm = math.inf
            best_point = None

            def grid_weights():
                if kk == 2:
                    for a in range(steps + 1):
                        yield [a / steps, (steps - a) / steps]
                else:
                    for a in range(steps + 1):
                        for b in range(steps + 1 - a):
                            yield [a / steps, b / steps,
                                   (steps - a - b) / steps]

            for w in grid_weights():
                q = compute_q(flat, offsets, rows, w)
                ll = log_likelihood(counts, q, N)
                if abs(ll - result["log_likelihood"]) <= 1e-9:
                    norm = sum(v * v for v in w)
                    if norm < best_norm:
                        best_norm, best_point = norm, w
            if best_point is not None:
                got = sum(v * v for v in result["representative"])
                if got > best_norm + 1e-3:
                    print("试验 %d FAIL: 网格最小范数 %g < 代表 %g"
                          % (trial, best_norm, got))
                    failures += 1

    print("\n随机交叉验证：%d 组，失败 %d 组" % (trials, failures))
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
