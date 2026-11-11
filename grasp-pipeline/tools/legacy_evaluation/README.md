# 原 v10 测评工具参考

这里的 11 个 Python 文件直接来自最终 130 任务交付的 `research-source.zip` 顶层，保持原样。它们与新 `src/` 运行模块分开，避免把调度、服务器准备或报告脚本当作策略核心。

| 类别 | 文件 |
| --- | --- |
| 原服务器运行环境准备 | `setup_remote.py`、`activate_runtime.py`、`extract_runtime.py`、`probe_runtime.py` |
| 测评调度与状态 | `launch.py`、`run_queue.py`、`status.py` |
| 整理结果和报告 | `finalize.py`、`package.py`、`render_report.py`、`validate_delivery.py` |

这些脚本保留原服务器路径、旧 `versions/v10` 布局和测评协议；没有被改写成新目录的一键评测工具，也不会由 `run.py` 默认执行。查看原测评复现过程时使用它们，整理版单任务运行使用根目录 `run.py`。后续评测和交付阶段再明确统一评测入口，不能直接在本目录执行旧队列来跑整理版。
