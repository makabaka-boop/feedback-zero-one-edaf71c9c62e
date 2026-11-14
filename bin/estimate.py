#!/usr/bin/env python3
"""命令行入口：python -I bin/estimate.py input.json output.json

成功才替换旧输出；任何校验或求解错误以非零状态退出，并打印
``字段路径: 原因`` 形式的定位信息，不产生输出文件改动。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from mixture_est.errors import ImpossibleEstimation, ValidationError
from mixture_est.io_ import dump_output, load_input
from mixture_est.solver import estimate
from mixture_est.validation import validate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="混合模型后端比例估计（手写 EM + 单纯形/主动集）"
    )
    parser.add_argument("input", help="输入 JSON 路径")
    parser.add_argument("output", nargs="?", help="输出 JSON 路径（缺省打印到标准输出）")
    args = parser.parse_args(argv)

    try:
        raw = load_input(args.input)
        validated = validate(raw)
        try:
            result = estimate(validated)
        except ImpossibleEstimation:
            result = {"status": "impossible"}
    except ValidationError as exc:
        print("输入非法 -> %s" % exc, file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("读取/解析失败 -> %s" % exc, file=sys.stderr)
        return 2

    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        dump_output(args.output, result)
        print("已写入 %s（status=%s）" % (args.output, result["status"]),
              file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
