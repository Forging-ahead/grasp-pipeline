# 基线核验与运行说明

## 本地核验

从工作区根目录执行：

```bash
python3 pipeline-organization/tools/inventory_baseline.py --check
```

该命令只读、不联网、不使用 GPU，也不导入策略或仿真依赖。它检查最终结果 ZIP 指纹、研究源码 ZIP CRC、全部交付文件内容指纹、50 个基线副本、39 个原开发 Python 文件、任务身份、固定初态和结果统计。

不带 `--check` 时工具会创建尚不存在的冻结副本并更新本目录 `audit/` 清单；遇到已有副本内容不一致会失败，不覆盖该文件。它不会修复或重写原始实验产物。独立 HTML 若被改名，只接受一个内容 SHA256 完全相同的同目录 HTML，并记录新旧名称。

[VALIDATION.json](../audit/VALIDATION.json) 是本阶段核验结果。清单核验与静态语法解析通过，不等于已在本机重新运行模型或重现成功轨迹。

## 历史测评的启动方式

以下是最终测评队列使用的命令形态，用于说明实际入口；本阶段未再次执行。

```bash
SIM_PY=/mnt/public/changwei/Graspgen/libero-final130-four4090-20261111/runtime/openpi/bin/python
V10_ROOT=/mnt/public/changwei/Graspgen/libero-final130-four4090-20261111/versions/v10
"$SIM_PY" "$V10_ROOT/run_task.py" \
  libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket \
  --gpu 0 --tag full130 --init 0
```

原 `full130` 输出目录已经存在，程序会拒绝复用。新实验应从本次 50 文件基线建立独立工作副本，准备环境、修改独立部署路径、使用新的版本及 tag；不要直接在冻结参考内继续调参。

`run_task.py` 会从自身旁边读取 `ALL_TASKS.json`，输出到自身目录的 `runs/<tag>/<task_key>/`。它不是一个通用 benchmark CLI。换文件位置时要一起保留任务清单和 `bridge/`，换依赖路径时要形成新的配置记录。

## 运行前提

1. 准备 Linux NVIDIA CUDA 与 EGL 运行条件，以及相互匹配的三个进程环境。
2. 准备 LIBERO 任务、初态、资产，GraspGen 及 Panda 权重，cuRobo 源码和机器人资产；核对 [配置与依赖](CONFIG_AND_RUNTIME.md) 的版本及指纹。
3. 核对任务清单、驱动覆盖、随机种子、初态与动作和墙钟预算。新增部署路径不能直接冒充原配置哈希。
4. 在独立输出目录做环境和坐标诊断、FK 校准及真实碰撞检查。旧 `doctor` 入口只证明相应环境与几何检查，不等于已跑通模型和完整任务。
5. 单任务验证通过后再决定是否运行更大回归。本轮没有启动这些实验。

第一阶段不提供自动环境安装器或一键重跑 130 任务的承诺。旧队列和运行环境准备脚本保留在研究源码 ZIP 顶层，包含原服务器路径与调度逻辑；后续整理时应作为测评工具单独处理。

## 如何读取一次运行

| 文件 | 用途 |
| --- | --- |
| 根 `RESULT.json` | 整个任务状态、原始目标、阶段状态、最终成功、步数和耗时 |
| `environment/ENVIRONMENT.json`、`environment/config.json` | 实际环境、初态来源、机器人、相机和创建环境的配置 |
| `SKILL_PLAN.json`、`PACKING_PLAN.json` | 目标阶段计划和多物体放置布局 |
| `stage_*/config.json`、`stage_*/RESULT.json` | 对应阶段的实际参数和细分结果 |
| `perception.json`、`input_points.npz` | 抓取输入来源及实际点云，通常位于抓放阶段子目录 |
| `grasp_candidates.npz`、`selected.json` | 模型候选、分数和最终选择，按阶段保存 |
| `plans/`、`selection/` 与各类诊断 JSON | 规划、预检、接触和控制拒绝原因，按阶段保存 |
| `trajectory.npz`、`states.json`、`rollout.mp4` | 连续执行记录；失败时保留已产生部分 |

交付目录的 `cases/<task_key>/` 另含 `ANALYSIS.json`、`KEYPOINTS.json`、`video.mp4` 等打包产物。这些不是新的物理执行。

不要只看进程退出码或某个子工具的 `success`。完整管线口径为：根任务的 `pipeline_success` 为真、`official_goal_at_end` 为真且执行动作数大于零。指令顺序与轨迹质量还要另行检查。规划失败时即使没有动作，也应记录规划尝试，不能计为完成。

## 后续回归记录

后续整理必须记录新源码和配置哈希、解释器及模型版本、任务和初态、实际规划后端、原生任务成功、管线完成、阶段失败原因与视频。若行为或参数改变，建立新版本，保留旧 v10 结果，不覆盖它来更新成功率。
