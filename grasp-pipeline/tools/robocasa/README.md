# RoboCasa 工具索引

从运行目录 `grasp-pipeline/` 执行。任务入口统一使用 `robocasa.py`；以下工具负责资源、诊断和独立实验，不代替主策略。可先对相应脚本使用 `--help`。

| 用途 | 工具 |
| --- | --- |
| 固定版本资产获取与校验 | `download_assets.py` |
| 原生环境/机器人/任务绑定检查 | `probe_env.py`、`probe_robot.py`、`probe_bindings.py` |
| 几何与控制核验 | `check_geometry_control.py` |
| 只读初态/失败放置几何扫描 | `diagnose_case.py`、`diagnose_placement.py` |
| 开发和主实验矩阵、前置证据 | `run_matrix.py`、`build_readiness.py` |
| 同初态候选池配对复测 | `run_pairs.py` |
| 矩阵/迁移报告 | `build_report.py`、`build_v10_report.py` |
| 65 项冻结覆盖 | `run_atomic65.py`、`atomic65.py`、`atomic65_case.py`、`atomic65_driver.py` |
| 65 项统计、媒体/轨迹审核和进度 | `report_atomic65.py`、`audit_atomic65.py`、`watch_atomic65.py` |
| r13 同初态 22 项修复复跑 | `run_pickplace22.py`、`pickplace22_records.py`、`report_pickplace22.py` |
| 成功轨迹 teacher 关键点 | `export_keypoints.py` |

## 复现本次 65 项覆盖

使用包含固定 RoboCasa 依赖的仿真解释器；四卡示例：

```bash
python tools/robocasa/run_atomic65.py --paths configs/robocasa/paths.local.json --output /data/atomic65-r10-seed0 --gpus 0 1 2 3 --sessions-per-gpu 2
# 上一步只生成冻结计划；实际执行必须使用另一个全新输出目录。
python tools/robocasa/run_atomic65.py --paths configs/robocasa/paths.local.json --output /data/atomic65-r10-run-001 --gpus 0 1 2 3 --sessions-per-gpu 2 --execute
python tools/robocasa/audit_atomic65.py --root /data/atomic65-r10-run-001 --output /data/atomic65-audit-001.json --require-complete
python tools/robocasa/report_atomic65.py --root /data/atomic65-r10-run-001 --output /data/atomic65-r10-run-001.html
```

该工具明确固定为 r10 actor 指纹、pretrain、一个 seed、5000 步采集预算，22 项动作尝试和 43 项场景检查。它会拒绝不同 actor；新策略应建立新实验版本，不删除指纹检查冒充原版。GPU 列表可按本机资源调整，每卡会话数为 1 或 2；新机器并发能力需自行验证。

报告内嵌可用视频，可以单文件转发；未支持任务只包含真实初态。规划等待不推进仿真，因此短视频不代表已经尝试闭合抓取。视频、HDF5、报告和输出目录均不进入源码 Git。

## 结果解释

r13 使用独立的 22 项复跑入口，不能通过修改 r10 指纹继续称为 r10。它从原完整矩阵读取同一任务绑定，首次动作前验证全部初始物理数组；使用全新 GraspGen 候选，固定 5000 步采集预算与 1200 秒墙钟上限。`report_pickplace22.py --watch` 只读进度；完整结束后使用 `--audit` 与 `--output` 审核并生成含新旧 44 段视频的离线 HTML。条件、修复及开发批次排除记录见 [r13 复跑记录](../../docs/robocasa/R13_RETEST.md)。

只读几何扫描不执行任务；配对候选池复测不重新推理 GraspGen。这些结果不能与新模型推理的完整连续回合混为一组。关键点来自 code-policy teacher，未经观测充分性审核的导出保持 `ready_for_sft=false`。

完整条件与已知限制见[65 项记录](../../docs/robocasa/ATOMIC65_RUN.md)、[继承与预算说明](../../docs/robocasa/V10_MIGRATION.md)和[分支入口](../../docs/robocasa/README.md)。
