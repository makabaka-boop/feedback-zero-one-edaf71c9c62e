#!/usr/bin/env bash
# 验收入口：隔离模式运行全部测试，不依赖第三方库与当前目录
set -euo pipefail
cd "$(dirname "$0")"
echo "== 验收测试 =="
python3 -I tests/test_acceptance.py
echo
echo "== 随机交叉验证（独立坐标上升/网格核对） =="
python3 -I tests/test_random_crosscheck.py
