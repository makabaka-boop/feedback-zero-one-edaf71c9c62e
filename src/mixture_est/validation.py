"""输入 JSON 的严格校验。

全部响应矩阵元素按精确分数（:class:`fractions.Fraction`）解析，
列和为 1 的校验因此是精确的，不引入浮点误差。
"""

import re
from fractions import Fraction

from .errors import ValidationError

# p/q 字符串：十进制非负整数 / 正整数，不接受小数、空白、符号
_FRACTION_RE = re.compile(r"^[0-9]+/[0-9]+$")
# 非空 ASCII 字母数字标识
_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


def _check_identifier(value, path):
    if not isinstance(value, str):
        raise ValidationError("标识必须为非空 ASCII 字母数字字符串", path)
    if not _ID_RE.match(value):
        raise ValidationError("标识必须为非空 ASCII 字母数字字符串", path)


def _parse_fraction(value, path):
    if not isinstance(value, str) or not _FRACTION_RE.match(value):
        raise ValidationError(
            "元素必须为 p/q 形式的字符串（分子非负整数、分母正整数）", path
        )
    numerator_text, denominator_text = value.split("/")
    numerator = int(numerator_text)
    denominator = int(denominator_text)  # 正则保证全数字
    if denominator == 0:
        raise ValidationError("分母必须为正整数", path)
    result = Fraction(numerator, denominator)
    if result < 0 or result > 1:
        raise ValidationError("元素必须落在 [0,1] 内", path)
    return result


def _parse_loss_budget(value):
    """解析可选 loss_budget：非负 p/q 字符串（分子非负整数、分母正整数）。

    与响应矩阵元素不同，预算不要求落在 [0,1] 内；非法一律定位到
    ``loss_budget`` 并整次拒绝。
    """
    if not isinstance(value, str) or not _FRACTION_RE.match(value):
        raise ValidationError(
            "loss_budget 必须为 p/q 形式的字符串（分子非负整数、分母正整数）",
            "loss_budget",
        )
    numerator_text, denominator_text = value.split("/")
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator == 0:
        raise ValidationError("loss_budget 分母必须为正整数", "loss_budget")
    result = Fraction(numerator, denominator)
    if result < 0:
        raise ValidationError("loss_budget 必须非负", "loss_budget")
    return result


def validate(raw):
    """校验并归一化原始输入。

    :returns: ``{"classes": [...], "experiments": [...], "loss_budget":
              Fraction|None}``，其中 ``A`` 的元素为 :class:`Fraction`，
              ``n`` 为整数元组；未提供 loss_budget 时其值为 ``None``。
    :raises ValidationError: 任何非法结构、重复、未知成员或数值。
    """
    if not isinstance(raw, dict):
        raise ValidationError("根节点必须为对象", "$")

    allowed_top = {"classes", "experiments", "loss_budget"}
    unknown = set(raw) - allowed_top
    if unknown:
        raise ValidationError("未知字段 %r" % sorted(unknown)[0], sorted(unknown)[0])
    for key in ("classes", "experiments"):
        if key not in raw:
            raise ValidationError("缺少必填字段", key)

    loss_budget = None
    if "loss_budget" in raw:
        loss_budget = _parse_loss_budget(raw["loss_budget"])

    classes_raw = raw["classes"]
    if not isinstance(classes_raw, list) or len(classes_raw) == 0:
        raise ValidationError("classes 必须为非空数组", "classes")

    classes = []
    seen = set()
    for index, value in enumerate(classes_raw):
        path = "classes[%d]" % index
        _check_identifier(value, path)
        if value in seen:
            raise ValidationError("类别标识重复", path)
        seen.add(value)
        classes.append(value)

    experiments_raw = raw["experiments"]
    if not isinstance(experiments_raw, list) or len(experiments_raw) == 0:
        raise ValidationError("experiments 必须为非空数组", "experiments")

    experiments = []
    seen_ids = set()
    n_classes = len(classes)

    for e_index, experiment in enumerate(experiments_raw):
        base = "experiments[%d]" % e_index
        if not isinstance(experiment, dict):
            raise ValidationError("实验必须为对象", base)

        unknown_fields = set(experiment) - {"id", "A", "n"}
        if unknown_fields:
            field = sorted(unknown_fields)[0]
            raise ValidationError("未知字段", "%s.%s" % (base, field))
        for field in ("id", "A", "n"):
            if field not in experiment:
                raise ValidationError("缺少必填字段", "%s.%s" % (base, field))

        id_path = "%s.id" % base
        _check_identifier(experiment["id"], id_path)
        if experiment["id"] in seen_ids:
            raise ValidationError("实验 id 重复", id_path)
        seen_ids.add(experiment["id"])

        matrix_raw = experiment["A"]
        if not isinstance(matrix_raw, list) or len(matrix_raw) == 0:
            raise ValidationError("A 必须为行数非零的数组（响应矩阵）", "%s.A" % base)

        matrix = []
        for i, row in enumerate(matrix_raw):
            row_path = "%s.A[%d]" % (base, i)
            if not isinstance(row, list) or len(row) != n_classes:
                raise ValidationError(
                    "A 的每行长度必须与 classes 数量一致（%d）" % n_classes, row_path
                )
            parsed_row = []
            for j, element in enumerate(row):
                parsed_row.append(_parse_fraction(element, "%s[%d]" % (row_path, j)))
            matrix.append(parsed_row)

        # 每列和精确等于 1
        for j in range(n_classes):
            column_sum = sum((matrix[i][j] for i in range(len(matrix))), Fraction(0))
            if column_sum != 1:
                raise ValidationError(
                    "A 的第 %d 列（零基）和必须为 1，实际为 %s" % (j, column_sum),
                    "%s.A[*][%d]" % (base, j),
                )

        counts_raw = experiment["n"]
        n_rows = len(matrix)
        if not isinstance(counts_raw, list) or len(counts_raw) != n_rows:
            raise ValidationError(
                "n 必须为与 A 行数等长（%d）的数组" % n_rows, "%s.n" % base
            )

        counts = []
        total = 0
        for i, count in enumerate(counts_raw):
            count_path = "%s.n[%d]" % (base, i)
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValidationError("计数必须为非负整数（禁止布尔值）", count_path)
            if count < 0:
                raise ValidationError("计数必须为非负整数", count_path)
            counts.append(count)
            total += count
        if total <= 0:
            raise ValidationError("每个实验的计数总数必须为正", "%s.n" % base)

        experiments.append(
            {"id": experiment["id"], "A": matrix, "n": tuple(counts)}
        )

    return {
        "classes": classes,
        "experiments": experiments,
        "loss_budget": loss_budget,
    }
