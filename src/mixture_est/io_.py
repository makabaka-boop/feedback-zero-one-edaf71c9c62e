"""JSON 读写与原子替换。"""

import json
import os
import tempfile


def load_input(path):
    # parse_constant 把 NaN/Infinity 等非法 JSON 常数显式拒绝
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("非法 JSON 常数: %s" % value)
            ),
        )


def dump_output(path, payload):
    """先写同目录临时文件，成功后原子替换旧输出。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".estimate-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      allow_nan=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
