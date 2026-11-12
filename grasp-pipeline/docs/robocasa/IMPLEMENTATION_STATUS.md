# RoboCasa 接入状态与运行方法

更新日期：2026-11-12。此分支是开发中的接入版本，LIBERO v10 源码、130 任务配置及冻结基线未改动。下文区分已写入代码、已实际验证和仍待验证的能力。

当前版本为 `robocasa-port-r13-v10`。沿用成熟 v10 的核心函数，修正抬升前持物登记与刷新、源支撑交接、最多 4 个放置朝向、相同无效起点复用及失败结果保存。同初态完整重跑原 22 项后，完整成功仍为 4/22，原生预算内成功由 2 项降为 1 项；不能声称整体改善或稳定迁移。详见 [r13 修复复跑](R13_RETEST.md) 和 [机读结果](../../migration/robocasa/VALIDATION_PICKPLACE22_20261112.json)。

原 r10 的 65 项覆盖（22 项执行、43 项场景检查）独立保留；本轮未重跑或启用其余 43 项。r13 每项仍使用同一 5000 步采集预算，总墙钟上限从 900 调回 1200 秒。全部初态及版本核对通过，22 段新视频与轨迹通过媒体和对齐检查；这是一次固定初态工程对照，不是多初态标准评测。
之前三个原生预算回归为 0/3，另有相同候选池与重新推理的两条额外成功轨迹（730/710 步），属于独立的开发验证，没有并入本轮 65 项统计。最终 104 项 RoboCasa 检查与 121 项旧检查合计 225 项通过。源码继承关系与局限见[v10 迁移记录](V10_MIGRATION.md)。下文十回合 0/10 属于 r7 历史结果。

## 已增加的代码

| 部分 | 实现与证据边界 |
| --- | --- |
| 独立入口与配置 | `robocasa.py` 的 list/config/probe/validate/run；12 个内置语义任务绑定，65 项覆盖工具另有 22 个显式共享抓放绑定；机器路径与策略参数分离 |
| 原生环境 | 固定 RoboCasa commit，校对原生任务、split、horizon；单一连续回合；保存 XML、物理状态、RNG、原始成功函数与版本 |
| 感知 | 同一物理状态渲染 MuJoCo 实例掩码；RGB-D 可见像素转世界系点云；记录相机内外参、可见像素和来源；无完整几何补点 |
| 机器人与执行 | 从 composite controller 读取动作切片和单位；保留 6D 末端控制；实测 gripper-root 到 TCP 变换；真实动作计数与接触监测 |
| 规划 | 复用 GraspGen、cuRobo 隔离 worker；新 robosuite 资产生成 Panda 模型；20 点 FK 校验；独立 MjData 做 IK 和碰撞预检；持物附件；cuRobo 图搜索回退、关节目标撤回及双模型检查后的路径简化 |
| 抓放策略 | 候选筛选、接近、闭合、双侧接触、抬升保持、目标区域候选、搬运、降低、释放、撤离及原生目标判定 |
| 机构与移动原型 | 读取机构关节轴和锚点，生成分段目标；底盘 A* 与原生速度控制；复合子目标调度。默认开关关闭，尚未通过厨房物理验收 |
| 数据与实验 | N 个动作与 N+1 个观测对齐；HDF5、视频、阶段目标、实际后端、失败分类；E1/E2 矩阵、冻结指纹、主实验前置证据检查 |
| 报告与标签 | 全部计划回合保留分母；不同版本不计成功；完整矩阵的 Wilson 区间；HTML 可内嵌视频；关键点导出明确为 code-policy teacher，默认不能直接用于 SFT |
| 资源获取 | 固定官方 RoboCasa/NVIDIA Hugging Face 资产版本，校验发布的 SHA256；安全解压和逐文件清单；不同内容的已有文件不覆盖 |

运行实现位于 `src/graspgen_libero_bridge/{adapters/robocasa,planning,control,policy,runtime}`，工具位于 `tools/robocasa/`。旧 LIBERO actor 和配置没有接入新的分支判断。

## r7 首轮接入验证（历史记录）

- 新隔离环境：Python 3.11.10，robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`，MuJoCo 3.3.1，NumPy 2.2.5；RoboCasa 源码固定为 `3443cdf3dd20f9ae8ed41adba442059b506891d0`。
- PandaOmron 在 robosuite Lift 中通过 20 个 FK 姿态校验：最大位置误差约 `7.08e-16 m`、旋转误差约 `4.86e-16 rad`。空载位置、旋转与夹爪开闭诊断实际执行了 41 步。
- `grasp-diagnostic-007` 在固定 seed 0 的 robosuite Lift 中完成真实 RGB-D → GraspGen → cuRobo → 接近 → 闭合 → 抬升，360 个物理动作，原生 Lift 成功判定为 true，代码指纹在运行中保持一致。**这是机器人链路诊断，不是 RoboCasa 厨房任务，也不计入 E1/E2。**
- 原生 RoboCasa `PickPlaceCounterToStove` 已通过环境创建、reset、RGB-D/实例掩码和关闭检查。原生厨房空载控制连续两轮通过，每轮 12 个位置/旋转目标、146 个真实动作；第二轮最大位置误差约 7.08 mm、旋转误差约 3.32°。
- r7 使用原生动作增量系数 1.0 后再次完成 12 个空载目标及夹爪验证，共 143 步。`PickPlaceCounterToStove / pretrain / seed 1` 的连续回合 `native-pick-place-011` 完整成功：547 步、241.93 秒、原生最终目标为 true、完整管线为 true、运行期间代码指纹一致。视频为左机位、右机位、腕部相机三个同步视角。
- 上述成功回合导出 4 个 teacher 关键点：`execute_grasp@0`、`lift_object@188`、`place_object@229`、`open_gripper@417`。它们是 code-policy teacher 标签，尚未通过学生观测充分性审核，`ready_for_sft=false`。
- 新增回归检查最近一次为 **64 passed**，涵盖动作切片、坐标、旋转、真实步数和退出、隔离预检、分割与深度、机构几何、导航、资源校验、超时清理、不混版本统计、IK 分支衔接、规划模型检查、软限位向内恢复、路径简化、原生目标稳定结束及关键点的学生输入字段隔离。原有 LIBERO 回归 **121 passed**。
- LIBERO 静态保真核对通过：39 个原始 Python 文件、1,040 组有效配置比较均保持一致。这不是一次新的 LIBERO GPU 轨迹测评。

已定位并修复的接入问题：Kitchen 不接收 `camera_segmentations` 参数；几何依赖在 EGL 相机启动后导入会损坏后续画面；初始折叠姿态存在保守球模型误报；缓存默认写入已满系统盘。初态恢复使用经过完整 MuJoCo 预检的实际动作，不直接修改真实关节状态，不关闭碰撞约束。

厨房与物体资产的 90 个压缩包均已下载、解压并通过 SHA256/逐文件清单校验。12 个声明任务均做过原生环境/绑定检查：首轮 11 项通过，`ArrangeBreadBasket` 的开放式柜架无门关节问题修正后单独复验通过。这些是零动作接入检查，不能计为任务成功。

当前已有一条完整原生厨房抓放成功演示。它是额外的开发诊断，不能并入后面的冻结矩阵。r7 按两个开发抓放任务各 5 个初态完成同版十回合验证，**完整成功为 0/10，原生最终成功为 0/10**：

| 任务 | 初态 | 完整成功 | 视频 |
| --- | --- | --- | --- |
| `PickPlaceCounterToStove` | pretrain seed 0–4 | 0/5 | 5/5 已保留并解码检查 |
| `PickPlaceSinkToCounter` | pretrain seed 0–4 | 0/5 | 5/5 已保留并解码检查 |

6 回合在闭合前未找到可执行候选，2 回合闭合后抬升的起点被 cuRobo 判为与原物体容器冲突，1 回合最后接近路径被真实几何预检拒绝，1 回合放下后撤离的关节路径优化失败。它们不能统称为 GraspGen 抓不住物体。全部 10 回合的代码/策略指纹一致且运行中未变；0 动作回合仅保留真实初始画面。十回合的 95% Wilson 区间为 0–27.75%，不支持稳定迁移的结论。

额外成功演示与矩阵中相同 seed 1 的复测结果不同，不能挑选成功演示替换失败回合。当前尚未通过 M2 的跨初态验收。没有执行 35 回合全技能开发矩阵或 240 回合 target 主实验；实测的主实验前置检查为未通过。摘要证据见[本轮验证记录](../../migration/robocasa/VALIDATION_20261112.json)，各模块与文件见[改动清单](FILE_CHANGES.md)。

下一轮先检查同一候选从接近、闭合到抬升的完整衔接，提前排除 cuRobo 起点冲突；复查真实跟踪后的接近路径偏差；再处理撤离规划失败后的共享备选目标和有限图搜索。六个候选搜索失败仍需区分终点不可达、模型过度保守和搜索预算不足，现有结果不证明这些场景无解。修复后使用新版本完整重跑，保留本轮 0/10 作为独立记录。

早期同版 r1 的三个开发诊断完整成功为 0/3：两个在抓取前候选规划失败，一个抓起后因 cuRobo 球模型认为手臂接近柜门而拒绝全部搬运起点。r2 在相同 seed 1 初态中提前检出局部抬升的模型冲突，回退到通过两套检查的 cuRobo 抬升路径，消除了该起点错误，但默认放置终点未通过规划。失败状态副本中的几何扫描找到 45°/90° 朝向、5 cm 净空的可行 IK 终点；r3 因此采用有限的共享高度候选。几何 IK 通过不等于路径或任务成功，各版结果分别保留，不合并最优结果。

后续定位并处理了：直线撤离扫到柜体、末端目标 IK 选择折叠手腕分支、MuJoCo 约 0.00019 rad 的软限位偏差被当成硬错误，以及长路径/保守动作增量导致耗尽 600 步。观测状态始终保留原值；软限位路径只允许在 0.001 rad 内单调向合法范围恢复，优化种子/控制参考保留 0.01 rad 内侧余量。原生 horizon、成功函数、碰撞监测没有放宽。撤回使用当前回合记录的初始关节姿态作为 cuRobo 关节目标；原生目标连续稳定 1 秒后可结束可选的回位路径，记录 `path_completed=false`，不伪称到达了完整回位终点。

同 seed 1 复测的 scene XML、初始 qpos 和目标点云哈希已核对一致，但 GraspGen 候选池仍出现跨进程差异，原因尚未完成定位。因此版本间的单次成功/失败不是模块消融证据。运动策略与参数的开发涉及 `PickPlaceCounterToStove`、`PickPlaceSinkToCounter`；全部 12 类都做过源码与接入检查，其中开放式柜架规则由 `ArrangeBreadBasket` 的绑定检查引出。`not_used_for_tuning` 只能理解为未用于运动参数调试，不能称完全未接触的任务。

## 运行

配置预览不导入仿真或模型：

```bash
python3 robocasa.py list
python3 robocasa.py config PickPlaceCounterToStove --seed 0
python3 robocasa.py run PickPlaceCounterToStove --output /data/runs/preview --dry-run
```

复制 `configs/robocasa/paths.example.json` 到被 Git 忽略的 `paths.local.json`，填写独立仿真解释器、固定 RoboCasa 源码、新 robosuite 资产、原有 GraspGen/cuRobo 环境与 Panda 权重的绝对路径。不要在 LIBERO 仿真环境里升级 robosuite、MuJoCo 或 NumPy。

官方 Box 下载不可用时，可使用固定版本的官方数据仓库：

```bash
python tools/robocasa/download_assets.py --robocasa-repo /data/robocasa --cache /data/asset-cache
# 若官方站点不可达，可显式选择 --endpoint https://hf-mirror.com
```

资源脚本依赖 `huggingface_hub`，不导入仿真。资产版本与校验清单写入 `ASSET_LOCK.json`；目录存在不代表下载完成。数据遵循上游许可，不进入本代码仓库。

在仿真解释器可用且资产就绪后，每次使用新的输出目录：

```bash
python3 robocasa.py probe PickPlaceCounterToStove --paths /data/paths.local.json --output /data/runs/probe-001
python3 robocasa.py validate PickPlaceCounterToStove --execute --paths /data/paths.local.json --output /data/runs/control-001
python3 robocasa.py run PickPlaceCounterToStove --seed 0 --gpu 0 --paths /data/paths.local.json --output /data/runs/pick-place-001
```

`probe` 只检查原生生命周期；`validate` 默认只做 CPU FK，增加 `--execute` 才有空载动作。机器人单独诊断另有 `tools/robocasa/probe_robot.py`，不能替代厨房任务验收。

```bash
python3 tools/robocasa/run_matrix.py --experiment E1_pick_place_development --output /data/runs/pick-place-preview
# 当前共享抓放实现的两个开发任务 × 5 个初态；不是“未修改策略的最小移植”基线
python3 tools/robocasa/run_matrix.py --experiment E1_development_snapshot --output /data/runs/e1-preview
# 加 --execute --paths ... --gpus 0 1 才实际执行 35 回合
python3 tools/robocasa/build_readiness.py --development-root /data/runs/e1 --control-validation /data/runs/control-001 --output /data/runs/READINESS.json
python3 tools/robocasa/run_matrix.py --experiment E2_main --paths /data/paths.local.json --readiness /data/runs/READINESS.json --gpus 0 1 --output /data/runs/e2 --execute
python3 tools/robocasa/build_report.py --root /data/runs/e2 --output /data/reports/robocasa.html
python3 tools/robocasa/export_keypoints.py --help
```

E2 只有在同版原生开发证据和控制验收通过后才能执行。未运行回合会显示待运行，报告不会把它们描述成已测失败或最终成功率。未实现/关闭的技能明确返回 `UNSUPPORTED_SKILL`，视频可以只有初始画面，不能解释为模型夹取失败。

## 仍未完成的计划项

1. 原生厨房抓放跨初态稳定性与机构/移动/复合技能的物理验收；自动为抓放调整底盘和躯干；在通过验证前不开放相关默认开关。
2. 将新 driver 的内部提供方 API 完整绑定到公开九类 Protocol 和版本化 Schema。目前不能宣称九类接口验收已完成。
3. E3 同候选池重放与严格初态配对、E4 点云对照。已有 `diagnose_case.py` 的零动作初态几何扫描和 `diagnose_placement.py` 的失败后物理状态副本扫描；它们不恢复完整控制器/任务内部状态，不构成物理回放或成功轨迹。现有几何候选/随机排序/局部 IK 开关尚不是完成的对照实验。
4. 初态可恢复性、相机几何残差与接触稳定性审计，以及组件耗时在所有失败出口上的完整统计。
5. 35 个全技能开发回合与冻结后 240 个 target 回合；目前只进行抓放开发验证。目标可观测性、学生输入中的特权字段隔离、工具标签层级和 teacher 关键点审核前，`training_candidate`、`ready_for_sft` 保持 false。

r7 使用了共享的 8 cm 接近、15 cm 抬升、10/5/15 cm 搬运净空候选及 16 次组合上限。r10 已将抓放主链改为原 v10 搜索与控制规则，具体有效参数见 `libero_v10_rules` 和上述迁移记录。任务 JSON 仍只绑定对象、fixture 和阶段顺序，不保存逐任务/seed 的世界坐标或成功轨迹。
