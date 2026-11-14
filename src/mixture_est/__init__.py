"""后端比例估计（混合模型 MLE）自写实现。

模块组成：
    errors     结构化校验错误
    validation 输入 JSON 的严格校验与精确分数解析
    em         EM（乘性迭代）求 MLE，含 Anderson 加速
    polytope   仿射多面体上的单纯形 LP 与主动集最小范数 QP（全部手写）
    solver     主编排：可行性、MLE、代表解、比例范围与见证
    io_        JSON 读写与原子替换
"""

from .solver import estimate, ImpossibleEstimation
from .errors import ValidationError

__all__ = ["estimate", "ImpossibleEstimation", "ValidationError"]
