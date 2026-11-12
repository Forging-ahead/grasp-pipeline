# RoboCasa 分支入口

本分支基于 `libero` 的成熟 v10 实现，增加 RoboCasa 原生环境、任务、机器人模型和控制绑定。**抓放核心直接调用或继承原函数，不重新写一套简化策略。** 原函数及共享参数的实际复用关系见[v10 迁移记录](V10_MIGRATION.md)；新 benchmark 的开发要求见[基线继承要求](../integration/BASELINE_FIRST.md)。

## 当前能做什么

最新 r10 固定初态覆盖检查：65 项中，22 项实际运行抓放策略，4 项完整成功、14 项策略失败、4 项超时；另外 43 项尚未支持，只创建场景并保存观测。不能将它们称为已执行任务后失败。

本轮显式采用 5000 步采集预算，4 项成功中 2 项在原生 horizon 内完成。每任务仅一个初态，不代表多初态平均成功率。65 段视频和轨迹全部通过解码/对齐检查。详见[覆盖条件、任务与失败原因](ATOMIC65_RUN.md)及[机读结果](../../migration/robocasa/VALIDATION_ATOMIC65_20261112.json)。

机构、旋钮、按钮、底盘、复合操作和多体持物仍是未启用原型；咖啡机出水口对位规则尚缺。改变开关不能代替任务绑定与真实物理验证。teacher 使用仿真实例掩码、几何与位姿，尚不等价于仅 RGB 感知或真机部署。

## 组件位置

| 位置（相对运行目录） | 责任 |
| --- | --- |
| `robocasa.py`、`apps/robocasa/` | 配置预览、探针、验证与连续回合入口 |
| `adapters/robocasa/` | 原生生命周期、任务/区域、机器人状态、RGB-D/实例掩码与记录 |
| `policy/robocasa_v10.py`、`adapters/robocasa/v10_preview.py` | 适配原 v10 候选、接触、搬运和放置函数所需输入 |
| `planning/robocasa_*.py` | cuRobo 及局部规划包装、机器人模型、持物和预检 |
| `control/robocasa_*.py` | 原 v10 控制绑定、原生动作、反馈与预算 |
| `runtime/robocasa_*.py` | 配置入口、执行生命周期、来源指纹、候选重放与进程清理 |
| `configs/robocasa/` | 策略、内置语义任务和机器路径示例；见[配置说明](../../configs/robocasa/README.md) |
| `tools/robocasa/` | 资产、诊断、实验、报告和关键点；见[工具索引](../../tools/robocasa/README.md) |
| `tests/robocasa/` | RoboCasa 回归；原 LIBERO 测试保持原位置 |

前三类源码目录位于 `src/graspgen_libero_bridge/`；`policy`、`planning`、`control`、`runtime` 均在该包内。详细列表见[改动清单](FILE_CHANGES.md)。

## 准备和运行

从仓库根目录进入 `grasp-pipeline/`。先按[独立环境说明](../../deploy/robocasa/README.md)准备固定版本的 RoboCasa、robosuite、资产和隔离解释器。不要升级原 LIBERO 环境。

```bash
cp configs/robocasa/paths.example.json configs/robocasa/paths.local.json
# 编辑七项绝对路径；本机配置不提交 Git。
python3 robocasa.py list
python3 robocasa.py config PickPlaceCounterToStove --seed 0 --paths configs/robocasa/paths.local.json
python3 robocasa.py run PickPlaceCounterToStove --seed 0 --paths configs/robocasa/paths.local.json --output /data/runs/preview --dry-run
```

上述预览不启动模型或仿真。准备好仿真解释器后，在该解释器中执行探针与控制验收，再运行任务：

```bash
python robocasa.py probe PickPlaceCounterToStove --paths configs/robocasa/paths.local.json --output /data/runs/probe-001
python robocasa.py validate PickPlaceCounterToStove --execute --paths configs/robocasa/paths.local.json --output /data/runs/control-001
python robocasa.py run PickPlaceCounterToStove --seed 0 --gpu 0 --paths configs/robocasa/paths.local.json --budget-profile libero-v10-teacher --output /data/runs/collect-001
```

每次使用新输出目录。默认预算模式是 `native`；示例显式选择 `libero-v10-teacher` 才扩到 5000 步，原生 horizon 和原生预算内成功仍单列。内置 CLI 任务目录为 12 项；65 项覆盖工具另有显式语义绑定，不会自动出现在 `robocasa.py list` 中。

## 检查与资料

在有 NumPy、SciPy、MuJoCo、pytest 和 `contracts/requirements-check.txt` 所列 jsonschema 的开发解释器中运行 `python -m pytest -q tests`。资源安装测试会检查临时目录剩余空间；空间紧张时将 `TMPDIR` 和 pytest `--basetemp` 指向数据盘。静态保真与文档检查仍使用 `tools/verify_preservation.py`、`tools/check_integration_docs.py`，不需要 GPU。

- [接入状态与待完成项](IMPLEMENTATION_STATUS.md)
- [接入规划](PLAN.md)与[实验设计](EXPERIMENT_PROTOCOL.json)
- [最新 65 项覆盖结果](ATOMIC65_RUN.md)
- [开发诊断与继承关系](V10_MIGRATION.md)

Git 只提交源码、配置模板、测试、文档和结果摘要；权重、完整轨迹、视频、报告 HTML、环境和缓存独立保存。历史 r7/r10 诊断与 65 项覆盖结果分别留档，不合并成功率。
