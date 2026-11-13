"""错误类型。"""


class ValidationError(ValueError):
    """输入结构或数值非法。

    ``path`` 为定位字段路径，数组使用零基下标，例如
    ``experiments[0].A[1][2]``、``classes[3]``。
    """

    def __init__(self, message, path):
        self.path = path
        full_message = "%s: %s" % (path, message)
        super().__init__(full_message)


class ImpossibleEstimation(Exception):
    """所有可行权重都使似然为负无穷（正计数落在全零响应行）。"""
