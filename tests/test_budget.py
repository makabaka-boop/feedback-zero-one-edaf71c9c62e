"""loss_budget 功能验收测试。

覆盖（题目要求的独立解析样例与小维对拍）：
  * 题目给定解析样例（二阶单位阵、n=[1,0]、预算 1/1）；
  * 独立解析样例（n=[3,1]、预算 1/2，手工二分求根对拍）；
  * 零预算退化为原范围、省略字段向后兼容；
  * 原任务 impossible 时仍仅返回原状态；
  * 多实验共享比例、零计数、完全相同响应列、边界端点；
  * 见证似然阈值、归一化、端点一致性；
  * 非法 loss_budget 一律定位到 loss_budget 并整次拒绝；
  * 小维（2/3 类）密集网格独立对拍。

运行：python3 -I tests/test_budget.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from mixture_est.em import compute_q, log_likelihood
from mixture_est.errors import ImpossibleEstimation, ValidationError
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


def _flatten(validated):
    return _prepare(validated)[2:]


def _witness_likelihood(result, validated, witness):
    _, _, flat, offsets, rows, counts, total_n = _prepare(validated)
    return log_likelihood(
        counts, compute_q(flat, offsets, rows, witness), total_n
    )


def _assert_budget_structure(result, validated, budget_value):
    """budget_ranges 结构、端点、见证合法性的通用核对。"""
    classes = validated["classes"]
    n_classes = len(classes)
    threshold = result["log_likelihood"] - budget_value
    check("含 budget_ranges 字段", "budget_ranges" in result)
    br = result["budget_ranges"]
    check("budget_ranges 数量", len(br) == n_classes)
    for j, item in enumerate(br):
        check("class 原序 %d" % j, item["class"] == classes[j])
        check("端点次序 %s" % item["class"],
              item["minimum"] <= item["maximum"] + TOL)
        check("min 见证达到端点 %s" % item["class"],
              approx(item["min_witness"][j], item["minimum"]))
        check("max 见证达到端点 %s" % item["class"],
              approx(item["max_witness"][j], item["maximum"]))
        for key in ("min_witness", "max_witness"):
            w = item[key]
            check("见证长度 %s" % item["class"], len(w) == n_classes)
            check("见证归一化 %s" % item["class"],
                  approx(sum(w), 1.0, 1e-9))
            check("见证非负 %s" % item["class"], all(v >= -1e-9 for v in w))
            check("见证有限 %s" % item["class"],
                  all(math.isfinite(v) for v in w))
            ll = _witness_likelihood(result, validated, w)
            check("见证似然不低于阈值 %s %s" % (item["class"], key),
                  math.isfinite(ll) and ll >= threshold - L_TOL,
                  "L=%.10f t=%.10f" % (ll, threshold))


def test_spec_example():
    print("\n[1] 题目解析样例：单位阵、n=[1,0]、预算 1/1")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "A",
             "A": [["1/1", "0/1"], ["0/1", "1/1"]], "n": [1, 0]}
        ],
        "loss_budget": "1/1",
    }
    result = estimate(validate(data))
    _assert_budget_structure(result, validate(data), 1.0)
    br = result["budget_ranges"]
    check("a 最小值 exp(-1)", approx(br[0]["minimum"], math.exp(-1)))
    check("a 最大值 1", approx(br[0]["maximum"], 1.0))
    check("b 最小值 0", approx(br[1]["minimum"], 0.0))
    check("b 最大值 1-exp(-1)",
          approx(br[1]["maximum"], 1.0 - math.exp(-1)))
    # 原报告字段原样保留
    check("原 ranges 保留", "ranges" in result and len(result["ranges"]) == 2)
    check("原 representative 保留", "representative" in result)


def test_independent_analytic_example():
    print("\n[2] 独立解析样例：n=[3,1]、预算 1/2，独立二分求根对拍")
    n1, n2, total = 3, 1, 4
    l_star = (3 * math.log(0.75) + math.log(0.25)) / total
    threshold = l_star - 0.5

    def f(w1):
        return (n1 * math.log(w1) + n2 * math.log(1 - w1)) / total - threshold

    roots = []
    prev = None
    for i in range(1, 100000):
        x = i / 100000.0
        value = f(x)
        if prev is not None and prev[1] * value < 0:
            lo, hi = prev[0], x
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        prev = (x, value)

    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e",
             "A": [["1/1", "0/1"], ["0/1", "1/1"]], "n": [n1, n2]}
        ],
        "loss_budget": "1/2",
    }
    result = estimate(validate(data))
    _assert_budget_structure(result, validate(data), 0.5)
    br = result["budget_ranges"]
    check("a 最小值=解析小根", approx(br[0]["minimum"], roots[0]))
    check("a 最大值=解析大根", approx(br[0]["maximum"], roots[1]))
    check("b 最小值互补", approx(br[1]["minimum"], 1 - roots[1]))
    check("b 最大值互补", approx(br[1]["maximum"], 1 - roots[0]))


def test_zero_budget_degenerates():
    print("\n[3] 零预算退化为原范围（0/1 与 0/5）")
    data = {
        "classes": ["a", "b", "c"],
        "experiments": [
            {"id": "e1",
             "A": [["1/2", "1/2", "0/1"], ["1/2", "1/2", "1/1"]],
             "n": [40, 60]},
            {"id": "e2",
             "A": [["1/1", "0/1", "1/4"], ["0/1", "1/1", "3/4"]],
             "n": [30, 70]},
        ],
    }
    base = estimate(validate(data))
    for zero in ("0/1", "0/5", "0/1000"):
        d = dict(data)
        d["loss_budget"] = zero
        r = estimate(validate(d))
        check("零预算 %s 与 ranges 完全一致" % zero,
              r["budget_ranges"] == base["ranges"])
        # 逐字段数值核对
        for j in range(3):
            check("零预算 min 端点一致 %s" % zero,
                  approx(r["budget_ranges"][j]["minimum"],
                         base["ranges"][j]["minimum"]))


def test_omitted_field_compatible():
    print("\n[4] 省略 loss_budget 时输出与旧版完全兼容（无该字段）")
    data = {
        "classes": ["x", "y"],
        "experiments": [
            {"id": "e",
             "A": [["1/3", "1/3"], ["2/3", "2/3"]], "n": [10, 20]}
        ],
    }
    result = estimate(validate(data))
    check("无 budget_ranges 字段", "budget_ranges" not in result)
    check("保留全部旧字段",
          all(k in result for k in
              ("status", "classes", "N", "log_likelihood",
               "representative", "ranges", "fitted_probabilities")))


def test_impossible_unchanged():
    print("\n[5] 原任务 impossible 时即使带预算也仅返回原状态")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1",
             "A": [["1/1", "1/1"], ["0/1", "0/1"]], "n": [0, 3]}
        ],
        "loss_budget": "1/1",
    }
    raised = False
    try:
        estimate(validate(data))
    except ImpossibleEstimation:
        raised = True
    check("仍抛 ImpossibleEstimation", raised)


def test_identical_columns_full_range():
    print("\n[6] 完全相同响应列：似然与 w 无关，预算范围内取满 [0,1]")
    data = {
        "classes": ["x", "y"],
        "experiments": [
            {"id": "e",
             "A": [["1/3", "1/3"], ["2/3", "2/3"]], "n": [10, 20]}
        ],
        "loss_budget": "1/2",
    }
    result = estimate(validate(data))
    _assert_budget_structure(result, validate(data), 0.5)
    for item in result["budget_ranges"]:
        check("%s 范围 [0,1]" % item["class"],
              approx(item["minimum"], 0.0) and approx(item["maximum"], 1.0))


def test_multi_experiment_shared_weights():
    print("\n[7] 多实验共享比例：budget 范围包住 MLE 范围，网格对拍")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1",
             "A": [["1/2", "1/4"], ["1/2", "3/4"]], "n": [10, 10]},
            {"id": "e2",
             "A": [["3/5", "1/10"], ["2/5", "9/10"]], "n": [5, 15]},
        ],
        "loss_budget": "1/2",
    }
    validated = validate(data)
    result = estimate(validated)
    _assert_budget_structure(result, validated, 0.5)
    for base, bud in zip(result["ranges"], result["budget_ranges"]):
        check("budget 下界不高于 MLE 下界",
              bud["minimum"] <= base["minimum"] + TOL)
        check("budget 上界不低于 MLE 上界",
              bud["maximum"] >= base["maximum"] - TOL)


def test_zero_counts_boundary():
    print("\n[8] 零计数 + 边界 MLE：见证全部满足阈值且端点被真正达到")
    data = {
        "classes": ["p", "q"],
        "experiments": [
            {"id": "e",
             "A": [["3/4", "1/2"], ["1/4", "1/2"]], "n": [50, 0]}
        ],
        "loss_budget": "1/4",
    }
    validated = validate(data)
    result = estimate(validated)
    _assert_budget_structure(result, validated, 0.25)
    # 端点不应只在原最优解（MLE 是 w=[1,0] 单点）：预算下 min 必须 < 1
    check("预算下 p 的最小值离开 MLE 单点",
          result["budget_ranges"][0]["minimum"] < 1.0 - 1e-4)


def test_boundary_endpoints_vertices():
    print("\n[9] 边界端点：大预算下端点可抵达单纯形 0/1（有限似然处）")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e",
             "A": [["1/2", "1/4"], ["1/2", "3/4"]], "n": [3, 7]}
        ],
        "loss_budget": "100/1",
    }
    result = estimate(validate(data))
    br = result["budget_ranges"]
    check("大预算 a 取满 [0,1]",
          approx(br[0]["minimum"], 0.0) and approx(br[0]["maximum"], 1.0))
    check("大预算 b 取满 [0,1]",
          approx(br[1]["minimum"], 0.0) and approx(br[1]["maximum"], 1.0))


def test_invalid_budget_rejected():
    print("\n[10] 非法 loss_budget 一律定位到 loss_budget 并整次拒绝")
    base_experiment = {
        "id": "e", "A": [["1/1"]], "n": [1]
    }
    illegal = ["0.5", "1", "-1/2", "2", "1/-2", "/2", "1/", "",
               "x/y", " 1/2", "1 /2", "1/0", 5, True, None, 1.5, ["1/2"]]
    for bad in illegal:
        data = {
            "classes": ["a"],
            "experiments": [base_experiment],
            "loss_budget": bad,
        }
        try:
            validate(data)
            check("拒绝 %r" % bad, False, "未抛异常")
        except ValidationError as exc:
            check("拒绝 %r 且路径=loss_budget" % bad,
                  exc.path == "loss_budget", exc.path)
    # 合法预算（含非单位分数、大数、零）被接受
    for good in ("0/1", "1/2", "3/2", "100/7", "999999999/1"):
        validated = validate({
            "classes": ["a"],
            "experiments": [base_experiment],
            "loss_budget": good,
        })
        check("接受合法预算 %s" % good,
              validated["loss_budget"] is not None
              and validated["loss_budget"] >= 0)


def _random_column_stochastic(rng, rows, cols):
    matrix = [[None] * cols for _ in range(rows)]
    for j in range(cols):
        cuts = sorted(rng.randrange(1, 1000) for _ in range(rows - 1))
        bounds = [0] + cuts + [1000]
        for i in range(rows):
            matrix[i][j] = "%d/1000" % (bounds[i + 1] - bounds[i])
    return matrix


def _grid_extremes(validated, threshold, steps):
    _, _, flat, offsets, rows, counts, total_n = _prepare(validated)
    n_classes = len(validated["classes"])
    lows = [math.inf] * n_classes
    highs = [-math.inf] * n_classes

    def scan(weights):
        q = compute_q(flat, offsets, rows, weights)
        value = log_likelihood(counts, q, total_n)
        if math.isfinite(value) and value >= threshold - 1e-9:
            for j in range(n_classes):
                lows[j] = min(lows[j], weights[j])
                highs[j] = max(highs[j], weights[j])

    if n_classes == 2:
        for a in range(steps + 1):
            scan([a / steps, (steps - a) / steps])
    elif n_classes == 3:
        for a in range(steps + 1):
            for b in range(steps + 1 - a):
                scan([a / steps, b / steps,
                      (steps - a - b) / steps])
    return lows, highs


def test_small_dimension_grid_crosscheck():
    print("\n[11] 小维（2/3 类）密集网格独立对拍")
    rng = random.Random(20261113)
    total_cases = 0
    worst = 0.0
    for trial in range(16):
        n_classes = rng.choice([2, 3])
        classes = ["c%d" % j for j in range(n_classes)]
        experiments = []
        for e in range(rng.randint(1, 2)):
            n_rows = rng.randint(2, 4)
            matrix = _random_column_stochastic(rng, n_rows, n_classes)
            counts = [rng.randint(0, 20) for _ in range(n_rows)]
            if sum(counts) == 0:
                counts[0] = 1
            experiments.append(
                {"id": "e%d" % e, "A": matrix, "n": counts}
            )
        for budget_text in ("1/10", "1/2", "1/1"):
            budget = float(__import__("fractions").Fraction(budget_text))
            data = {
                "classes": classes,
                "experiments": experiments,
                "loss_budget": budget_text,
            }
            validated = validate(data)
            try:
                result = estimate(validated)
            except ImpossibleEstimation:
                continue
            total_cases += 1
            threshold = result["log_likelihood"] - budget
            steps = 4000 if n_classes == 2 else 60
            grid_lo, grid_hi = _grid_extremes(validated, threshold, steps)
            for j in range(n_classes):
                item = result["budget_ranges"][j]
                # 容差取网格分辨率的若干倍：2 类 1/4000=2.5e-4，
                # 3 类 1/60≈1.67e-2；求解器本身对细网格的偏差在 3e-5 内
                tolerance = 6e-4 if n_classes == 2 else 0.04
                d_lo = abs(item["minimum"] - grid_lo[j])
                d_hi = abs(item["maximum"] - grid_hi[j])
                worst = max(worst, d_lo, d_hi)
                check("网格对拍 trial=%d j=%d 下界" % (trial, j),
                      d_lo <= tolerance, "%.2e" % d_lo)
                check("网格对拍 trial=%d j=%d 上界" % (trial, j),
                      d_hi <= tolerance, "%.2e" % d_hi)
    print("  （对拍用例数 %d，最大偏差 %.2e）" % (total_cases, worst))


def main():
    tests = [
        test_spec_example,
        test_independent_analytic_example,
        test_zero_budget_degenerates,
        test_omitted_field_compatible,
        test_impossible_unchanged,
        test_identical_columns_full_range,
        test_multi_experiment_shared_weights,
        test_zero_counts_boundary,
        test_boundary_endpoints_vertices,
        test_invalid_budget_rejected,
        test_small_dimension_grid_crosscheck,
    ]
    for test in tests:
        test()
    print("\n" + "=" * 60)
    if _failures:
        print("失败 %d 项：" % len(_failures))
        for name, detail in _failures:
            print("  - %s  %s" % (name, detail))
        return 1
    print("全部 loss_budget 验收测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
