# 配置与依赖

## 哪份配置生效

最终 130 任务的优先级为：

1. `baseline/v10/ALL_TASKS.json` 对应任务的 `row.config`。
2. `run_task.py` 的 `dict(...)` 显式覆盖，包括命令行 `--gpu`、`--init`。
3. 启动环境前暂把 `destination` 设为 `target_object`，避免历史占位目标不存在；任务阶段再依据原始目标解析目的地。
4. 每阶段覆盖抓取对象、目的物体、目标区域和关系；多物体表面布局还可写入放置 XY 以及本阶段偏航候选。

源码中 `config.get(...)` 的默认值仅在该键未提供时生效。`bridge/configs/server.json` 是早期单任务示例，不是最终 130 任务的完整启动配置。各次运行的 `environment/config.json` 和 `stage_*/config.json` 保存当时实际配置，分别对应环境创建与阶段执行。

机器可读盘点见 [CONFIG_INVENTORY.json](../audit/CONFIG_INVENTORY.json)。配置读取位置见 [SOURCE_INVENTORY.json](../audit/SOURCE_INVENTORY.json)。这两份清单不会把源码中的所有数值常量都误称为可配置超参数。

## 最终驱动覆盖

以下为 `run_task.py` 第 20 至 25 行的实际覆盖值，未在本次整理中调整。

| 参数 | 最终值 | 作用 |
| --- | --- | --- |
| `simulation_gpu`、`model_gpu`、`planner_gpu` | 同一 `--gpu` | 单任务的三个进程使用同一张卡 |
| `init_state_id` | `--init`，本次为 0 | 官方初态索引 |
| `unsupported_skill` | `false` | 不沿用任务清单的旧“不支持”占位标记 |
| `max_steps` | 5000 | 管线执行动作预算，不能当作官方统一评测步数 |
| `composite_subgoal_execution` | `true` | 复合目标执行模式 |
| `geometric_placement_search`、`surface_release_search`、`open_container_release_search` | `true` | 放置目标搜索 |
| `container_xy_search`、`static_container_candidate_search` | `true` | 容器 XY 和静态候选搜索 |
| `minimum_downward_approach` | 0 | 候选方向筛选阈值 |
| `selection_rounds` | 6 | 候选筛选轮数 |
| `max_candidates_per_round` | 30 | 每轮候选上限 |
| `native_candidates_per_round` | 12 | 每轮原生规划候选上限 |
| `selection_timeout_s` | 180 | 每轮候选搜索预算，不是整个任务预算 |
| `place_tracking_extra_steps` | 240 | 放置控制允许的额外跟踪步数 |
| `place_tracking_extension_radius_m` | 0.015 | 跟踪延长适用的位置误差范围 |

驱动中的配置开关与源码是否读取它们应结合静态配置读取记录核对；本阶段保留实际值，不清理未使用字段。

## 任务清单中的共同设置

全部 130 个任务的清单配置均使用 `input_mode=rgbd`、`input_camera=agentview`、`camera_size=256`、`seed=0`、`settle_steps=10`、`num_grasps=200`。实际控制频率为 20 Hz。

共同参数包括：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `grasp_control_offset_m` | 0.097 m | GraspGen 夹爪参考点到控制 site 的轴向转换；需要按夹爪校准 |
| `lift_height_m` | 0.15 m | 抬升阶段控制 site 的增量高度，不是所有任务的统一放置高度 |
| `approach_distances_m` | `[0.1, 0.05]` m | 接近距离候选 |
| `transfer_heights_m` | `[0.12, 0.08, 0.04, 0.16, 0.2]` m | 搬运目标高度候选，不是一个固定终点 |
| `transfer_yaw_degrees` | `[0, 45, -45, 90, -90, 180]` 度 | 搬运朝向候选；每阶段可能进一步覆盖 |
| `plan_timeout_s` | 12 s | 单次规划请求预算 |
| `waypoint_max_steps` | 120 | 单路点跟踪预算 |
| `position_integral_gain`、`position_integral_bound_m` | 0.8、0.008 m | OSC 跟踪积分项 |
| `joint_reference_margin_rad`、`planning_joint_margin_rad` | 0.02、0.0001 rad | 执行参考和规划的限位余量 |
| `joint_recovery_steps`、`joint_recovery_reference_margin_rad` | 100、0.2 rad | 限位恢复预算和参考余量 |

其他配置如目标身份、区域和关系由任务而异。关节技能、接触容差、几何搜索和控制仍有源码默认值或常量，后续整理需要按“机器人参数、规划参数、技能参数、环境参数、测评预算”归档；本阶段未把它们全部外移为新配置。

测评队列另设墙钟预算：96 个任务 1200 秒，另外 34 个任务 1800 秒，包括加载、规划、渲染和控制。超时先发 SIGINT，随后按记录的等待期限结束进程。它与执行步数预算、单次规划预算互相独立。

## 三个独立进程环境

以下版本来自最终测评的运行记录，不是本机新安装结果，也不是完整依赖锁文件。

| 角色 | 记录的主要版本 | 工作 |
| --- | --- | --- |
| 仿真与控制主进程 | Python 3.11.10，NumPy 1.26.4，SciPy 1.16.2，PyTorch 2.6.0+cu124，MuJoCo 3.3.7，robosuite 1.4.1，OpenCV 4.11.0 | 环境、观测、局部 IK、执行、记录与判定 |
| GraspGen 子进程 | Python 3.11.10，PyTorch 2.6.0+cu124，OmegaConf 2.3.1 | 加载官方 Panda 生成器和判别器，输出候选与分数 |
| cuRobo 子进程 | Python 3.11.10，cuRobo 0.7.8，PyTorch 2.6.0+cu124，SciPy 1.17.1 | 机器人 FK、IK、碰撞与全局运动规划 |

精确解释器、模块路径、版本和文件指纹保存在 [DEPENDENCIES.json](../audit/DEPENDENCIES.json)，原证据在 [RUNTIME_VERSIONS.json](../../outputs/libero-final130-four4090-20261111/delivery/RUNTIME_VERSIONS.json)。版本匹配记录覆盖其中列出的模块，不等于完整 pip/系统/CUDA 驱动环境都已打包。

## 外部路径和模型

服务器公共前缀为 `/mnt/public/changwei/Graspgen`：

| 配置或组件 | 当前路径 |
| --- | --- |
| LIBERO checkout | `LIBERO/` |
| GraspGen fork checkout | `graspgen-curobo-collect/` |
| 实际 GraspGen 源码 | `graspgen-curobo-collect/third_party/GraspGen/` |
| 模型解释器 | `setup/grasp-model/bin/python` |
| 模型配置 | `graspgen-curobo-collect/models/checkpoints/graspgen_franka_panda.yml` |
| 规划解释器入口 | `setup/planner-python-wrapper`，实际环境为 `setup/planner/` |
| cuRobo 源码 | `setup/deps/curobo/src/` |
| 最终测评仿真解释器 | `libero-final130-four4090-20261111/runtime/openpi/bin/python` |

生成器 `graspgen_franka_panda_gen.pth` 与判别器 `graspgen_franka_panda_dis.pth` 由外部已安装模型目录提供。指纹分别为：

```text
generator     0597583b89b322d42ceb4e596967d6ed68d1b56cba4039895909ccd5bdc66eff
discriminator e47d703c63b54c2d11fbc1effd43898f251b4147250888541e3b16e9c0d19e1c
```

基线不包含权重、上游 checkout、全部资产和三个完整运行环境。`pyproject.toml` 只声明本包，不会安装上述运行依赖。后续部署需要各角色依赖清单、资产路径与启动检查，不能直接对本目录执行一次 pip 安装就声称可复现。

四卡主机的原 `/opt/venv/openpi` 为 Python 3.10；本次实验使用独立复制的 Python 3.11.10 环境，部分解释器原绝对路径通过别名解析。别名记录在原测评 `RUNTIME_ALIASES.json` 中，属于该服务器的运行事实，不应写成外部接入方必须照抄的部署步骤。
