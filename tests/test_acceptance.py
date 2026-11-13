"""验收测试：覆盖通道联合识别、零计数、相同响应列、不可行、非法输入。

运行：python3 -I tests/test_acceptance.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from mixture_est.errors import ImpossibleEstimation, ValidationError
from mixture_est.solver import estimate
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
    return len(a) == len(b) and all(
        approx(x, y, tol) for x, y in zip(a, b)
    )


def validate_result(result, classes, experiments):
    """通用结构与数值合法性核对。"""
    n_classes = len(classes)
    check("status=ok", result["status"] == "ok")
    check("classes 原序", result["classes"] == classes)
    check("N 为总样本数",
          result["N"] == sum(n for exp in experiments for n in exp["n"]))

    rep = result["representative"]
    check("代表解长度", len(rep) == n_classes)
    check("代表解非负", all(v >= -TOL for v in rep))
    check("代表解归一化", approx(sum(rep), 1.0, 1e-9))
    check("代表解为有限数", all(math.isfinite(v) for v in rep))

    # 最小范数代表：在全部 MLE 见证上 sum w^2 最小
    witnesses = [rep]
    for r in result["ranges"]:
        witnesses.append(r["min_witness"])
        witnesses.append(r["max_witness"])
    rep_norm = sum(v * v for v in rep)
    check("代表解最小二乘范数",
          all(rep_norm <= sum(v * v for v in w) + 1e-9 for w in witnesses),
          "rep_norm=%s" % rep_norm)

    # 范围与见证一致性
    for j, r in enumerate(result["ranges"]):
        check("range.class 原序 %d" % j, r["class"] == classes[j])
        check("min<=max %s" % r["class"], r["minimum"] <= r["maximum"] + TOL)
        check("min 见证达到最小值 %s" % r["class"],
              approx(r["min_witness"][j], r["minimum"]))
        check("max 见证达到最大值 %s" % r["class"],
              approx(r["max_witness"][j], r["maximum"]))
        for w in (r["min_witness"], r["max_witness"]):
            check("见证归一化 %s" % r["class"], approx(sum(w), 1.0, 1e-9))
            check("见证非负 %s" % r["class"], all(v >= -1e-9 for v in w))
            check("见证有限 %s" % r["class"], all(math.isfinite(v) for v in w))

    # 拟合概率按实验原序、行和为 1
    check("拟合概率实验数",
          len(result["fitted_probabilities"]) == len(experiments))
    for k, exp in enumerate(experiments):
        fp = result["fitted_probabilities"][k]
        check("实验 id 原序 %d" % k, fp["id"] == exp["id"])
        check("q 行数 %d" % k, len(fp["q"]) == len(exp["n"]))
        check("q 行和为 1 %d" % k, approx(sum(fp["q"]), 1.0, 1e-9))
        check("q 非负 %d" % k, all(v >= -1e-12 for v in fp["q"]))

    # 所有见证达到同一归一化似然
    from mixture_est.em import compute_q, log_likelihood
    flat, offsets, rows, counts = _flatten(experiments, n_classes)
    total_n = sum(counts)
    base_l = result["log_likelihood"]
    for w in witnesses:
        q = compute_q(flat, offsets, rows, w)
        check("见证似然一致",
              approx(log_likelihood(counts, q, total_n), base_l, L_TOL))


def _flatten(experiments, n_classes):
    from fractions import Fraction
    flat, offsets, rows, counts = [], [], [], []
    offset = 0
    for exp in experiments:
        offsets.append(offset)
        rows.append(len(exp["A"]))
        for row in exp["A"]:
            flat.extend(float(Fraction(x)) for x in row)
        counts.extend(exp["n"])
        offset += len(exp["A"])
    return flat, offsets, rows, counts


def test_spec_example_identifiable_zero_row():
    print("\n[1] 题目示例：相同响应列 + 零计数行，范围 [0,1]，代表 0.5/0.5")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1", "A": [["1/1", "1/1"], ["0/1", "0/1"]], "n": [3, 0]}
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    check("L=0", approx(result["log_likelihood"], 0.0, L_TOL))
    check("代表 [0.5,0.5]", vec_approx(result["representative"], [0.5, 0.5]))
    check("a 范围 [0,1]",
          approx(result["ranges"][0]["minimum"], 0.0)
          and approx(result["ranges"][0]["maximum"], 1.0))
    check("b 范围 [0,1]",
          approx(result["ranges"][1]["minimum"], 0.0)
          and approx(result["ranges"][1]["maximum"], 1.0))
    check("拟合概率 [1,0]",
          vec_approx(result["fitted_probabilities"][0]["q"], [1.0, 0.0]))


def test_spec_impossible():
    print("\n[2] 题目示例：正计数落全零行 -> impossible")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1", "A": [["1/1", "1/1"], ["0/1", "0/1"]], "n": [0, 3]}
        ],
    }
    raised = False
    try:
        estimate(validate(data))
    except ImpossibleEstimation:
        raised = True
    check("抛 ImpossibleEstimation", raised)


def test_joint_channel_identification():
    print("\n[3] 通道联合识别：两个实验联合后列满秩，MLE 唯一，范围退化为点")
    # 实验1：类别 a/b 不可分；实验2：把两者区分开
    data = {
        "classes": ["a", "b", "c"],
        "experiments": [
            {"id": "ch1",
             "A": [["1/2", "1/2", "0/1"], ["1/2", "1/2", "1/1"]],
             "n": [40, 60]},
            {"id": "ch2",
             "A": [["1/1", "0/1", "1/4"], ["0/1", "1/1", "3/4"]],
             "n": [30, 70]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    # 联合可辨识 -> 每个范围 min≈max，且见证都等于代表
    for j, r in enumerate(result["ranges"]):
        check("类别 %s 范围退化" % r["class"],
              approx(r["minimum"], r["maximum"], 1e-7),
              "%s..%s" % (r["minimum"], r["maximum"]))
        check("类别 %s 见证=代表" % r["class"],
              vec_approx(r["min_witness"], result["representative"], 1e-7))
    # 独立网格搜索交叉核对 MLE 目标
    best = -math.inf
    best_w = None
    flat, offsets, rows, counts = _flatten(data["experiments"], 3)
    from mixture_est.em import log_likelihood
    N = sum(counts)
    steps = 80
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            w = [i / steps, j / steps, (steps - i - j) / steps]
            from mixture_est.em import compute_q
            q = compute_q(flat, offsets, rows, w)
            value = log_likelihood(counts, q, N)
            if value > best:
                best, best_w = value, w
    check("网格搜索似然不超过 MLE",
          best <= result["log_likelihood"] + L_TOL,
          "grid=%s mle=%s" % (best, result["log_likelihood"]))
    check("网格最优点接近代表",
          vec_approx(best_w, result["representative"], 0.03),
          "grid=%s rep=%s" % (best_w, result["representative"]))


def test_identical_response_columns():
    print("\n[4] 完全相同响应列：两类不可区分，代表均分，范围互补且和为 1")
    data = {
        "classes": ["x", "y"],
        "experiments": [
            {"id": "e",
             "A": [["1/3", "1/3"], ["2/3", "2/3"]],
             "n": [10, 20]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    check("代表均分", vec_approx(result["representative"], [0.5, 0.5]))
    rx, ry = result["ranges"]
    check("x 范围 [0,1]",
          approx(rx["minimum"], 0.0) and approx(rx["maximum"], 1.0))
    check("y 范围 [0,1]",
          approx(ry["minimum"], 0.0) and approx(ry["maximum"], 1.0))
    # 端点见证互补
    check("min 见证互补",
          vec_approx(rx["min_witness"], [0.0, 1.0]))
    check("max 见证互补",
          vec_approx(rx["max_witness"], [1.0, 0.0]))


def test_zero_counts_and_boundary():
    print("\n[5] 零计数 + 结构性零：列满秩实验 MLE 唯一，范围退化为点")
    # 2x2 列随机矩阵、列满秩 -> 唯一 MLE；第二行零计数，MLE 把概率压向第一行
    data = {
        "classes": ["p", "q"],
        "experiments": [
            {"id": "e",
             "A": [["3/4", "1/2"], ["1/4", "1/2"]],
             "n": [50, 0]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    # 列满秩 -> 每个范围 min≈max，见证都等于代表
    rp, rq = result["ranges"]
    check("范围退化为点(p)", approx(rp["minimum"], rp["maximum"], 1e-7),
          "%s..%s" % (rp["minimum"], rp["maximum"]))
    check("范围退化为点(q)", approx(rq["minimum"], rq["maximum"], 1e-7))
    check("代表分量非负", all(v >= -1e-9 for v in result["representative"]))
    check("代表归一化", approx(sum(result["representative"]), 1.0, 1e-9))
    # 解析解：max 0.75 wp+0.5 wq s.t. wp+wq=1 -> wp=1, wq=0，q=[0.75,0.25]
    check("MLE wp=1", vec_approx(result["representative"], [1.0, 0.0]),
          str(result["representative"]))
    check("拟合 q=[.75,.25]",
          vec_approx(result["fitted_probabilities"][0]["q"], [0.75, 0.25]))
    expected_l = (50 * math.log(0.75)) / 50
    check("似然解析值", approx(result["log_likelihood"], expected_l, L_TOL))


def test_single_class():
    print("\n[6] 单类别：w 必须为 1，q 等于该列")
    data = {
        "classes": ["only"],
        "experiments": [
            {"id": "e", "A": [["2/5"], ["3/5"]], "n": [7, 3]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    check("w=1", vec_approx(result["representative"], [1.0]))
    check("范围 [1,1]",
          approx(result["ranges"][0]["minimum"], 1.0)
          and approx(result["ranges"][0]["maximum"], 1.0))
    check("q=列",
          vec_approx(result["fitted_probabilities"][0]["q"], [0.4, 0.6]))


def test_multi_experiment_shared_weights():
    print("\n[7] 多实验共享 w：拟合概率逐实验核对 A_e w")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1",
             "A": [["1/2", "1/4"], ["1/2", "3/4"]], "n": [10, 10]},
            {"id": "e2",
             "A": [["3/5", "1/10"], ["2/5", "9/10"]], "n": [5, 15]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    w = result["representative"]
    from fractions import Fraction
    for k, exp in enumerate(data["experiments"]):
        for i, row in enumerate(exp["A"]):
            expected = sum(float(Fraction(row[j])) * w[j] for j in range(2))
            check("q[%d][%d]=Aw" % (k, i),
                  approx(result["fitted_probabilities"][k]["q"][i],
                         expected, 1e-9))


def test_validation_errors():
    print("\n[8] 非法输入拒绝与字段路径定位")

    def expect_error(data, path_fragment):
        try:
            validate(data)
        except ValidationError as exc:
            ok = path_fragment in exc.path
            check("拒绝且路径含 %s" % path_fragment, ok,
                  "实际路径=%s" % exc.path)
            return
        check("拒绝且路径含 %s" % path_fragment, False, "未抛异常")

    base = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1",
             "A": [["1/2", "1/4"], ["1/2", "3/4"]], "n": [1, 1]}
        ],
    }

    # 重复类别
    bad = {"classes": ["a", "a"], "experiments": base["experiments"]}
    expect_error(bad, "classes[1]")

    # 非法标识（含非字母数字）
    bad = {"classes": ["a", "b-2"], "experiments": base["experiments"]}
    expect_error(bad, "classes[1]")

    # 空 classes
    expect_error({"classes": [], "experiments": base["experiments"]}, "classes")

    # 重复实验 id
    bad = {
        "classes": ["a", "b"],
        "experiments": [base["experiments"][0],
                        {"id": "e1",
                         "A": [["1/2", "1/2"], ["1/2", "1/2"]], "n": [1, 1]}],
    }
    expect_error(bad, "experiments[1].id")

    # 列和不为 1（第一列 1/2+1/2=1 正确，第二列 1/4+3/4=1 也正确；
    # 这里把第二行第二列改成 1/4，使第二列和为 1/2）
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/2", "1/4"], ["1/2", "1/4"]], "n": [1, 1]}
        ],
    }
    expect_error(bad, "A[*]")

    # 元素越界
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["3/2", "1/2"], ["0/1", "1/2"]], "n": [1, 1]}
        ],
    }
    expect_error(bad, "A[0][0]")

    # 非分数字符串
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["0.5", "1/2"], ["1/2", "1/2"]], "n": [1, 1]}
        ],
    }
    expect_error(bad, "A[0][0]")

    # 布尔计数
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/2", "1/2"], ["1/2", "1/2"]],
             "n": [True, 1]}
        ],
    }
    expect_error(bad, "n[0]")

    # 负计数
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/2", "1/2"], ["1/2", "1/2"]], "n": [-1, 1]}
        ],
    }
    expect_error(bad, "n[0]")

    # 实验总数为零
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/2", "1/2"], ["1/2", "1/2"]], "n": [0, 0]}
        ],
    }
    expect_error(bad, "experiments[0].n")

    # A 行数与 n 不等长
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/2", "1/2"], ["1/2", "1/2"]], "n": [1]}
        ],
    }
    expect_error(bad, "experiments[0].n")

    # 未知顶层字段
    expect_error({"classes": ["a"], "experiments": base["experiments"],
                  "extra": 1}, "extra")

    # 未知实验字段
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/2", "1/2"], ["1/2", "1/2"]],
             "n": [1, 1], "z": 0}
        ],
    }
    expect_error(bad, "experiments[0].z")

    # 分母为零
    bad = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e", "A": [["1/0", "1/2"], ["0/1", "1/2"]], "n": [1, 1]}
        ],
    }
    expect_error(bad, "A[0][0]")


def test_three_identical_columns():
    print("\n[9] 三个完全相同响应列：三等分代表，各自范围 [0,1]")
    data = {
        "classes": ["a", "b", "c"],
        "experiments": [
            {"id": "e",
             "A": [["1/3", "1/3", "1/3"], ["2/3", "2/3", "2/3"]],
             "n": [10, 20]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    check("三等分代表",
          vec_approx(result["representative"], [1 / 3, 1 / 3, 1 / 3]))
    check("代表二乘范数=1/3",
          approx(sum(v * v for v in result["representative"]), 1 / 3, 1e-9))
    for r in result["ranges"]:
        check("%s 范围 [0,1]" % r["class"],
              approx(r["minimum"], 0.0) and approx(r["maximum"], 1.0))


def test_uninformative_experiment():
    print("\n[10] 一个实验完全无信息（列全同），另一个实验锁定比例")
    data = {
        "classes": ["a", "b"],
        "experiments": [
            {"id": "e1",
             "A": [["1/2", "1/2"], ["1/2", "1/2"]], "n": [3, 7]},
            {"id": "e2",
             "A": [["1/1", "0/1"], ["0/1", "1/1"]], "n": [4, 6]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    # e2 是确定性通道，直接锁定 w=[0.4,0.6]，e1 不提供额外信息
    check("代表 [0.4,0.6]",
          vec_approx(result["representative"], [0.4, 0.6]))
    for r in result["ranges"]:
        check("%s 范围退化为点" % r["class"],
              approx(r["minimum"], r["maximum"], 1e-7))


def test_large_counts():
    print("\n[11] 大计数压力：数值稳定且有限")
    data = {
        "classes": ["a", "b", "c"],
        "experiments": [
            {"id": "e",
             "A": [["1/2", "1/4", "1/8"],
                   ["1/2", "1/2", "3/8"],
                   ["0/1", "1/4", "1/2"]],
             "n": [1000000, 2000000, 500000]},
        ],
    }
    result = estimate(validate(data))
    validate_result(result, data["classes"], data["experiments"])
    check("N 正确", result["N"] == 3500000)
    check("似然有限", math.isfinite(result["log_likelihood"]))


def main():
    tests = [
        test_spec_example_identifiable_zero_row,
        test_spec_impossible,
        test_joint_channel_identification,
        test_identical_response_columns,
        test_zero_counts_and_boundary,
        test_single_class,
        test_multi_experiment_shared_weights,
        test_validation_errors,
        test_three_identical_columns,
        test_uninformative_experiment,
        test_large_counts,
    ]
    for test in tests:
        test()
    print("\n" + "=" * 60)
    if _failures:
        print("失败 %d 项：" % len(_failures))
        for name, detail in _failures:
            print("  - %s  %s" % (name, detail))
        return 1
    print("全部验收测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
