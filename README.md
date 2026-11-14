# 后端比例估计（混合模型 MLE）

多个检测通道的误识别率不同，若把每个通道分别反演再平均，会得到互相矛盾的
群体比例。本服务把所有实验放到**共享的类别比例向量 `w`** 下统一做最大似然
估计，并刻画 MLE 不唯一时全体解的范围。

全部数值算法（EM、单纯形、主动集 QP、高斯消元）均为手写，仅依赖 Python
标准库，不调用任何统计拟合或优化求解器。

## 模型

- 类别 `classes`（记数量为 J），实验 `experiments`（记数量为 E）。
- 实验 `e` 的响应矩阵 `A_e`：行=观测类别，列按 `classes` 原序；元素为
  `[0,1]` 内的 `p/q` 字符串，每列和为 1（列随机矩阵）。
- 计数 `n_ei >= 0`，每个实验计数总数为正。
- 所有实验共享 `w_j >= 0`、`sum_j w_j = 1`，拟合概率
  `q_ei = sum_j A_eij * w_j`。
- 总样本数 `N = sum_ei n_ei`，最大化归一化对数似然

      L(w) = (1/N) * sum_ei n_ei * ln(q_ei(w))

  零计数项贡献 0；正计数遇到零概率时 L 为负无穷。

## 输出

成功（`status="ok"`）时返回：

| 字段 | 含义 |
| --- | --- |
| `N` | 总样本数 |
| `log_likelihood` | 最大归一化对数似然 L |
| `representative` | 全部 MLE 解中 `sum w_j^2` 最小的**唯一**代表（严格凸），按 `classes` 原序 |
| `ranges[j].minimum/maximum` | 类别 j 在**全部 MLE 解**上的最小/最大比例 |
| `ranges[j].min_witness/max_witness` | 分别达到最小/最大端点的完整 `w` 见证（各端点无需同时达到） |
| `fitted_probabilities[e].q` | 实验 e 的拟合概率，按实验原序、观测行原序 |

若对任意可行 `w` 都有正计数落在零概率上（正计数单元对应整行为零），
仅返回 `{"status": "impossible"}`。

比例、概率、范围、见证绝对误差 ≤ 1e-6，归一化似然差 ≤ 1e-8。

## 算法

1. **精确校验**：用 `fractions.Fraction` 解析 `p/q`，列和为 1 的判定是精确的；
   拒绝重复 id/类别、未知字段、布尔计数、缺测、非法结构与数值，错误带字段
   路径（数组零基下标），如 `experiments[0].A[1][2]`。
2. **可行性**：任一正计数单元在其实验中对应全零行则 `impossible`。
3. **MLE**：EM 乘性迭代 `w_j <- w_j g_j / N`（MM，似然单调不减），内嵌
   Type-I Anderson 加速处理相同响应列的慢收敛；收敛后做 KKT 驱动的主动集
   精修，区分**结构性零**（`g_j<N`，固定为 0）与**非可辨识方向**（`g_j=N`，
   保持自由）。
4. **MLE 集合**：MLE 的拟合概率 `q*` 唯一，故 MLE 集合是仿射多面体
   `P = {w : sum w=1, Mw=q*, w>=0}`。对增广矩阵做 RREF 取独立等式。
5. **代表解**：手写原始主动集法解严格凸 QP `min sum w_j^2` s.t.
   `C w = d, w >= 0`（解唯一）。每次在当前固定零集的仿射面上求精确
   最小范数点（行约化 + Gram 系统），越界则比率步把触零坐标并入
   零集；面内最优后再用手写单纯形 LP 验证投影变分刻画
   `min <w,v> >= ||w||^2 (v in P)` 作为最优性证书，不满足则沿线段
   精确线搜索严格下降并收缩零集。绝不把"可行但非最优"的点当作代表。
6. **比例范围**：对每个类别分别手写两阶段单纯形（Bland 规则）解
   `min/max w_j`，顶点最优解即端点见证。
7. **自洽核对**：代表与全部见证必须满足 MLE 等式、非负、归一化，且达到同一
   归一化似然；任一不符即报错，不产出结果。

## 用法

```bash
# 写文件（成功才原子替换旧输出，失败不改动）
python3 -I bin/estimate.py input.json output.json

# 打印到标准输出
python3 -I bin/estimate.py input.json
```

作为库：

```python
from mixture_est.validation import validate
from mixture_est.solver import estimate
from mixture_est.errors import ImpossibleEstimation

try:
    result = estimate(validate(raw_input))
except ImpossibleEstimation:
    result = {"status": "impossible"}
```

建议始终用 `python3 -I`（隔离模式）运行，避免工作目录中的同名模块被导入。

## 目录

```
src/mixture_est/
  validation.py  严格输入校验 + 精确分数解析
  linalg.py      手写高斯消元
  em.py          EM + Anderson 加速 + KKT 主动集精修
  polytope.py    RREF 零空间、两阶段单纯形 LP、主动集最小范数 QP
  solver.py      主编排与自洽核对
  io_.py         JSON 读写、原子替换、NaN/Infinity 拒绝
  errors.py      ValidationError（带路径）/ ImpossibleEstimation
bin/estimate.py  命令行入口
tests/           验收测试 + 随机交叉验证
examples/        题目示例与典型场景
run_tests.sh     一键验收
```

## 验收

```bash
./run_tests.sh
# 或
python3 -I tests/test_acceptance.py
python3 -I tests/test_representative_qp.py
python3 -I tests/test_random_crosscheck.py
```

覆盖：题目两个示例、通道联合识别、零计数与结构性零、完全相同响应列、
单类别、多实验共享权重、三相同列、无信息实验、大计数，以及十余类非法输入
的路径定位。代表解另有专项回归：两个手算小例（最优代表含零分量）、冗余
观测约束、相同响应列均分、单位矩阵唯一解、列同步重排不变性，并用精确
分数的支撑面枚举独立核对代表的可行性与最优性。随机交叉验证用独立的
坐标上升与密集网格核对 MLE 目标、拟合概率、最小范数代表与端点见证。
