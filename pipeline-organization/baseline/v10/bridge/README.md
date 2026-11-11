Fixed release 2026-11-09: see FIXES.txt for controller, joint-limit and result-schema changes. The 2026-11-08 validation records below are historical.

# GraspGen / cuRobo → LIBERO 独立桥接项目

本项目把已有 GraspGen Panda 权重和 cuRobo 规划器接入官方 LIBERO 的 MuJoCo / robosuite 环境。桥接代码、运行配置和结果保存在本项目；官方 LIBERO 与原 `graspgen-curobo-collect` 的源码、权重及共享配置无需修改。

当前实现面向**显式指定目标的、单个刚性物体、单一 `On(target, destination)` 目标的抓取放置**。默认场景是抓起盘子与小碗之间的黑碗，放到盘子上。目标物体、目的物体与任务名称由配置给出，执行过程为接近、夹取、抬升、搬运、放下、释放和撤离。执行器会核对任务原有目标谓词与配置是否一致，拒绝多谓词或其他关系。它还没有实现从任意语言指令推导完整计划，也没有覆盖 LIBERO 全部任务、铰接物体操作或多物体操作。

## 验证状态

2026-11-08，当前实现分别取得一条 `geometry` 和一条 `rgbd` 完整任务成功记录，使用同一个官方任务、固定初态 0、seed 0。这是两种输入配置的工程验证；尚未评估多个任务或多个初态的成功率。

| 检查 | 状态 |
|---|---|
| CPU 单元测试 | 22 项全部通过，无跳过，约 2.69 秒；含真实 MuJoCo 的 4 项 Cartesian 测试 |
| `doctor`：官方环境、固定初态、渲染、点云、场景导出 | `geometry` 与 `rgbd` 均通过 |
| MuJoCo / 生成机器人模型 / cuRobo 的多姿态 FK 对齐 | 两条完整运行均通过启动时 FK 验证 |
| `geometry` 模式完整抓取放置 | `run_005`：任务与抓取均成功；583 步，153.19 秒，抬升约 0.151 m |
| `rgbd` 模式完整抓取放置 | `rgbd_001`：任务与抓取均成功；567 步，160.38 秒，抬升约 0.161 m |
| 多个官方初态上的成功率 | 未评估 |

两条记录的任务均为下文所列的黑碗放盘子任务，官方最终目标为 `(On akita_black_bowl_1 plate_1)`。调试中的失败记录保留，不把这两条成功记录表述为总体成功率。此前在 Isaac 中取得的抓取结果不计入这里。实际结果以各次运行的 `RESULT.json` 为准。

## 运行

在本项目根目录运行。脚本默认使用已有的 `/opt/venv/openpi/bin/python` 作为 LIBERO 解释器，并直接通过 `PYTHONPATH` 加载桥接代码，无需先安装本项目。

先检查环境与输入几何：

```bash
bash run.sh doctor --config configs/server.json --input-mode geometry
```

再运行完整抓取放置：

```bash
bash run.sh run --config configs/server.json --input-mode geometry --init-state 0 --seed 0
```

使用相机深度与目标实例掩码：

```bash
bash run.sh doctor --config configs/server.json --input-mode rgbd
bash run.sh run --config configs/server.json --input-mode rgbd --init-state 0 --seed 0
```

`--output` 必须是尚不存在的目录，每次复跑使用新目录。省略它时，程序在本项目的 `runs/` 下生成带时间戳和随机后缀的目录。`doctor` 会创建环境、渲染、加载固定初态、采样校准姿态并导出点云和碰撞场景；它不加载 GraspGen / cuRobo 工作进程，也不执行完整抓取，不能代替完整运行验收。

如需指定其他已有的仿真解释器：

```bash
LIBERO_PYTHON=/absolute/path/to/python bash run.sh doctor --config configs/server.json
```

更多命令行参数：

```bash
bash run.sh --help
```

可覆盖的参数包括 `--config`、`--output`、`--input-mode`、`--init-state`、`--seed` 和 `--candidate-offset`。其他设置通过独立 JSON 配置调整。当前代码使用 CUDA 运行模型与规划，并使用 EGL 渲染；运行前应选择可用的物理 GPU 编号。

## 两种输入模式

| 模式 | GraspGen 收到什么 | 使用了哪些额外信息 |
|---|---|---|
| `geometry` | 从目标物体完整仿真几何表面采样的 2048 个世界坐标点 | 真值目标身份、完整物体几何；用于工程诊断 |
| `rgbd` | 从配置相机的深度图中，经目标实例掩码提取并反投影的 2048 个可见表面点 | 真值目标实例掩码、相机内外参；没有实现从 RGB 学习目标分割 |

两种模式均使用米制坐标，**GraspGen 实际使用 XYZ 点几何，不使用 RGB 图像特征**；RGB 图像用于执行记录。模型工作进程对点云减均值，推理后只恢复一次均值，返回世界坐标抓取姿态。当前 `rgbd` 在回合初始稳定后采集一次目标点云用于生成候选；执行控制仍读取机械臂状态，规划也会同步仿真真值场景。

两种模式的碰撞世界、目标状态、放置目标位置和放置几何仍来自仿真真值。因此 `rgbd` 仅指抓取候选的输入换为可见深度点云，完整系统仍使用真值物体几何和状态，不能直接作为与仅 RGB 策略处于相同观测条件的比较。具体抓取输入来源以 `perception.json` 和结果中的 `grasp_input_*` 字段区分。两条早期成功记录生成于元数据字段细化前，保留原始凭据；其输入范围按本节和对应感知记录解读。

## 依赖与进程隔离

完整运行使用三个独立进程：

| 进程 | 服务器默认解释器 / 依赖 | 职责 |
|---|---|---|
| LIBERO 主进程 | `/opt/venv/openpi/bin/python`；MuJoCo、robosuite、NumPy、SciPy、PyTorch、OpenCV | 创建官方任务、感知、短距离接触阶段规划、执行和评分 |
| GraspGen 工作进程 | `setup/grasp-model/bin/python`；已有官方 Panda 权重与 GraspGen | 生成候选并评分 |
| cuRobo 工作进程 | `setup/planner-python-wrapper`；已有 cuRobo / CUDA 环境 | 机器人 FK、IK、碰撞规划和持物几何 |

其中 `setup/` 的服务器前缀为 `/mnt/public/changwei/Graspgen/`，实际绝对路径以 `configs/server.json` 为准。仿真环境还需要 `trimesh` 支持部分几何处理；碰撞网格凸包导出使用 SciPy。主进程记录最终导入的 LIBERO 路径与版本，以免无意加载服务器旧的 `/opt/libero` fork。

进程之间通过自动生成鉴权信息的本机 socket 和 NPZ 文件交换数据，不需要申请云 API Key。依赖已有环境的原因是 GraspGen、cuRobo 和 LIBERO 的版本要求不同；不要把官方旧训练环境的整份依赖直接安装进现有模型或规划环境。本项目的 `pyproject.toml` 也不会自动安装这三套运行依赖。

LIBERO 的 `LIBERO_CONFIG_PATH` 被设置为当前运行目录中的 `libero_config/`，启动前写入独立 `config.yaml`，不会覆盖默认的 `~/.libero/config.yaml`。固定初态、任务 BDDL 与物体资源直接读取新官方仓库；抓取测试不要求下载人类示范数据。新版 PyTorch 读取官方 NumPy 初态文件时显式使用 `weights_only=False`，兼容处理放在桥接代码中。

## 混合规划分工

当前全局规划与短接触运动采用不同规划器，最终都交给默认 OSC 控制器执行：

| 阶段 | 规划器 | 检查方式 |
|---|---|---|
| 候选抓取的接近 / 下降、持物运输 | cuRobo | 从实际 MJCF 生成的机器人模型、完整场景和保守碰撞表示；执行前再做 MuJoCo 预检 |
| 抬升 `lift`、放置下降 `place`、撤离 `retreat` | 主进程中的 `cartesian.py` | 从当前实际关节状态出发，用 MuJoCo `mj_jacSite`、阻尼最小二乘 IK、关节限位、直线 / SLERP 小步和关节段加密，逐点检查真实 MuJoCo 接触 |

短接触规划沿用当前阶段的手指与目标接触、持物状态和支撑面许可。持物预演中，目标 free-joint 按测量到的夹爪相对关系同步，用于检查物体与环境碰撞；退出规划时还原原物理状态、速度、控制输入和暖启动数据。真实执行仍由 OSC 和物理夹持完成，规划预演不会推进环境时间或直接执行物体搬运。

这种分工适用于局部抬升、放下和撤离；短接触规划器不会搜索绕过障碍物的全局路径。它使用现有监测器的接触容差和采样检查，不能提供连续时间的无碰撞保证。每个阶段的 `CONTACT_POLICY.json` 和规划结果记录实际采用的规划器。

## 默认任务与成功判定

默认任务：

```text
suite:       libero_spatial
task:        pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
target:      akita_black_bowl_1
destination: plate_1
goal:        (On akita_black_bowl_1 plate_1)
```

程序从官方任务的 `.pruned_init` 中按 `init_state_id` 选择固定初态，并先执行 `settle_steps` 次全零动作；当前服务器配置为 10 步。随后的一次运行最多执行一个实际抓取尝试；候选生成、规划与碰撞预检中可以筛选多个候选。该计数保存在 `physical_attempts`。

`RESULT.json` 区分两个结果：

- `grasp_success`：当前工程诊断阈值为物体相对初态抬升至少 6 cm，在抬升后的保持阶段测量。它不是原 Isaac 接触力验收，也不是完整任务完成。
- `task_success`：释放、撤离和等待后调用任务原有 `env.check_success()`，由官方 BDDL 目标谓词决定。

`status="completed"` 表示执行流程走到了末尾，仍需查看 `task_success`；完整运行以任务成功退出码 0，失败退出码 1。`doctor_passed` 只表示环境与几何检查通过。固定一个初态成功也不能推出整个 suite 的成功率。

## 配置项

主配置为 `configs/server.json`。建议复制成独立配置后修改，不更改两个依赖仓库的配置。

| 配置项 | 作用 |
|---|---|
| `libero_repo` | 官方 LIBERO checkout 根目录 |
| `grasp_repo` | 已有 `graspgen-curobo-collect` 根目录，用于定位第三方 GraspGen 源码 |
| `graspgen_python`、`graspgen_config` | 模型工作进程解释器与官方 Panda 权重配置 |
| `planner_python` | cuRobo 工作进程解释器或包装脚本 |
| `simulation_gpu`、`model_gpu`、`planner_gpu` | 各进程使用的物理 GPU 编号；主进程同步配置 EGL |
| `task_suite`、`task_name` | 官方 suite 与完整任务名称 |
| `target_object`、`destination` | 显式指定的目标和目的物体实例名 |
| `init_state_id`、`seed` | 官方初态索引与采样 / 模型随机种子 |
| `input_mode`、`input_camera` | `geometry` 或 `rgbd`；深度输入相机默认 `agentview` |
| `camera_size` | 默认 256，也可提供 `[height, width]` |
| `max_steps`、`settle_steps` | 执行动作总预算与初态稳定步数；稳定步不计入执行预算 |
| `num_grasps` | 单次模型调用生成的候选数量，默认 200 |
| `minimum_downward_approach` | 进入规划前要求候选从上方接近的阈值；属于显式工程筛选 |
| `candidate_offset` | 跳过评分排序并经过接近方向筛选后的若干候选，用于诊断；默认 0 |
| `max_candidates_per_round`、`selection_rounds` | 每轮规划的候选数量和筛选轮数 |
| `plan_timeout_s` | 每次位姿规划的时间预算 |
| `max_robot_spheres`、`attachment_slots` | 可选的 cuRobo 机器人碰撞球与持物球数量预算；需遵守所装版本的核函数上限 |
| `grasp_control_offset_m` | 原生 GraspGen 夹爪原点到 robosuite 控制 site 的轴向偏移，默认 0.097 m |
| `lift_height_m` | 抬升阶段的控制 site 增量高度，默认 0.15 m |

更低层的环境设置还支持 `control_freq`、`camera_names`、`render_gpu_device_id`、`task_order_index`、`calibration_sample_count`、`planning_plane_extent_m`。更改机器人、控制器、任务类别或末端几何需要相应代码适配，不能仅替换几个配置值。当前放置策略仅检查并执行与显式配置一致的单一 `On` 目标；其他单一关系和多个谓词的组合也需要新的任务策略。

## 坐标、碰撞和执行限制

内部位姿统一为 4×4 世界坐标矩阵。机械臂状态按命名关节读取；控制位姿取实际 `gripper0_grip_site` 的位置和姿态。默认 robosuite 末端观测的位置与四元数来自不同参考点，桥接代码不直接拼接这两项。规划器由已安装 robosuite 的 Panda / PandaGripper MJCF 生成机器人模型，并通过至少 5 组 MuJoCo 姿态检查 FK；同时验证生成模型与 cuRobo 的结果。

GraspGen 原生 Panda 的接近轴为 `+z`、闭合轴为 `±x`。默认到控制 site 的转换是沿原生 `z` 偏移 0.097 m；实际仿真夹爪与模型训练夹爪仍有几何细节差异，换资产后需要重新检查。原 Isaac 仓库的 0.1034 m 偏移和绕 Z 旋转不能直接照搬。

碰撞处理包含以下明确近似和边界：

- 场景包含所有非机器人接触几何，目标单独导出；mesh 使用 MuJoCo compiled 顶点构建逐 geom 凸包，避免重复应用原始缩放或把凹网格误当作 MuJoCo 的碰撞表面。
- 球、圆柱和部分其他形状在规划器中使用包围盒；地面平面采用覆盖机器人工作空间的有限厚盒；机器人碰撞使用逐几何 AABB 的保守球覆盖。这些近似可能拒绝物理上可行的运动。
- 抓取下降允许与目标接触时，优化器会移除整个目标的世界碰撞项；执行前的 MuJoCo 路径预检和执行时接触监测进一步限制机械臂 / 手掌碰撞。当前接触检查有容差，也没有给出连续时间、精确网格的无碰撞证明。
- 被抓物体附着到规划器的保守几何表示，运输时参与碰撞检查；离开原支撑面和放置到目的支撑面时，会按阶段放宽对应支撑几何。结果与规划目录记录具体接触策略。
- 保持 LIBERO 默认 7 维 `OSC_POSE` 动作：3 维平移增量、3 维轴角增量和 1 维夹爪命令。原来的 9 维关节目标不能直接发送给环境。
- 规划器关节路径先转换为控制 site 路径，再由 OSC 闭环跟踪。Panda 的冗余自由度可能使实际关节轨迹偏离规划轨迹；代码记录 / 限制偏差并监测实际接触，但这些检查不等于严格复现原关节路径。

当前控制、预检和碰撞近似服务于最小抓取放置实现。结果应连同输入模式、固定初态、控制器、近似策略和检查记录一起解读。

## 结果目录

不同运行阶段可能只生成其中一部分；失败时保留已有诊断。

| 路径 | 内容 |
|---|---|
| `RESULT.json` | 状态、抓取 / 任务成功、耗时、步数、错误和执行检查 |
| `config.json`、`ENVIRONMENT.json` | 本次完整配置、官方任务、固定初态与依赖来源 |
| `libero_config/config.yaml` | 本次独立的 LIBERO 资源路径 |
| `merged_model.xml` | 完整合并后的 MuJoCo 模型 |
| `initial_scene.json` | 初始机器人状态和完整导出的碰撞世界 |
| `calibration_samples.json`、`MUJOCO_CALIBRATION_SAMPLES.json` | 多姿态 FK 校准样本 |
| `input_points.npz`、`perception.json` | 实际模型输入点云与感知来源 |
| `grasp_candidates.npz` 及同名 `.json` | 原始候选、分数、采样中心与模型来源 |
| `screened_candidates.json`、`selected.json`、`selection_rejections.json` | 接近方向筛选、最终选择和拒绝原因 |
| `geometry/collision_meshes/` | compiled 物体网格的逐 geom 凸包 OBJ 缓存 |
| `planner_config/` | 生成的机器人模型、碰撞表示与 CPU / cuRobo FK 验证 |
| `selection/`、`plans/` | 各候选与阶段的世界、接触策略、规划结果及 `PATH.npz` |
| `plans/{lift,place,retreat}/CARTESIAN_RESULT.json` | 短接触阶段的真实 MuJoCo IK、碰撞检查数量、状态还原与失败原因；对应 NPZ 在持物时另含预演物体位姿 |
| `model_worker/worker.log`、`planner_worker/worker.log` | 两个依赖工作进程的日志 |
| `trajectory.npz` | 机械臂 / 手指状态、控制位姿、7 维动作与动作阶段 |
| `states.json` | 每个记录时刻的完整状态 |
| `rollout.mp4`、`preview.png` | 双相机执行录像和阶段预览 |

轨迹按 N 个动作对应 N+1 个记录状态保存。这里的视频和状态记录不包含原 Isaac 采集器的完整八子步接触力数据集；`doctor` 也不会生成模型候选或规划器验证文件。

## 代码入口

- `env_adapter.py`：官方 LIBERO、独立配置、固定初态、机器人 / 物体状态与校准。
- `pointcloud.py`、`graspgen_worker.py`：几何 / RGB-D 输入与官方模型推理。
- `scene.py`、`planning_geometry.py`、`planner_worker.py`：实际场景、机器人模型、碰撞、规划与持物几何。
- `cartesian.py`：真实 MuJoCo 的短接触阶段 IK 规划及物理状态恢复。
- `frames.py`、`controller.py`：坐标转换、OSC 跟踪、预检、接触监测和录像。
- `evaluate.py`：当前显式目标抓取放置流程与官方任务判定。
- `rpc.py`：本机工作进程通信与生命周期。

当前默认任务可用于验证桥接链路。扩展到其他刚性物体任务时需要核对目标关系和放置策略；扩展到门、抽屉等关节物体时，还需要新增相应操作策略。

CPU 测试可在已有仿真环境中运行，无需创建 GPU / 渲染上下文：

```bash
/opt/venv/openpi/bin/python -m unittest discover -s tests -v
```

`tests/test_cartesian.py` 使用小型原生 MuJoCo 模型验证 site-Jacobian IK、真实碰撞拒绝、持物预演刚体关系及成功 / 异常后的状态恢复；不依赖 LIBERO 渲染和模型权重。缺少 MuJoCo 或 SciPy 时这四项会跳过，跳过不计为通过。
