"""小型稠密线性代数工具（手写高斯消元，不依赖第三方库）。"""


class SingularMatrix(Exception):
    """系数矩阵在当前容差下奇异。"""


def solve_linear(matrix, rhs, ridge=0.0):
    """带部分主元的高斯消元解 ``A x = b``。

    ``ridge`` 非零时给对角线加 ``ridge * I`` 做微小正则，
    用于处理理论上奇异但数值上可正则化的 KKT 系统。
    """
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    if ridge:
        for i in range(n):
            aug[i][i] += ridge

    scale = max(
        (abs(aug[i][j]) for i in range(n) for j in range(n)), default=1.0
    )
    pivot_tol = 1e-14 * max(1.0, scale)

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) <= pivot_tol:
            raise SingularMatrix("第 %d 列无有效主元" % col)
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]

    return [aug[i][n] / aug[i][i] for i in range(n)]
