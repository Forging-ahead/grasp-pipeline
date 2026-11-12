# 从成熟 LIBERO v10 迁移，而非重新简化策略

2026-11-12。最终版本为 `robocasa-port-r10-v10`，修正首轮 r7 未完整继承策略的问题。r7、r8、r9 结果分别保留。下文列出真实复用的原函数，以及最终同版的有限开发验证，不代表完整 benchmark 测评。

随后使用同一 actor 完成[65 项基础任务覆盖检查](ATOMIC65_RUN.md)：22 项执行共享策略，4 项完整成功（其中 2 项在原生 horizon 内）；43 项尚未支持，只检查场景。该批次与下文开发诊断独立统计，未挑选历史成功替换失败。

## 实际复用的原函数

| 原 v10 实现 | RoboCasa 调用方式 |
| --- | --- |
| `PickPlace.select_candidate` | `V10Selection` 直接继承，未重写；6 轮、每轮最多 30 个候选，接近距离 10/5 cm |
| `PickPlace.recover_native_descent` | 直接继承；全局/下探失败后，尝试经过真实几何检查的局部接近与下探 |
| `preview_grasp_contacts` | 原函数；选择前静态双侧接触检查，实际闭合后仍检查原生双侧持物 |
| `plan_cartesian` | 原函数；通过适配器操作独立 `MjData`，用于接触、抬升、放置与撤离 |
| `PickPlace.transfer_with_alternatives` | 直接继承；复用原有 5 个搬运高度和 6 个偏航角，不再共用 r7 的 16 次限制 |
| `orientation_candidates` / `opening_offsets` | 原函数；根据真实目标开口计算物体姿态和内部 XY 偏移 |
| `container_release_candidates` | 原函数；最低可行释放高度搜索及物体完整脚印检查 |
| `PickPlace.place_in_container_static` | 直接继承；先预检移动与下放两段，再执行选中的方案，并从实测状态重算最终下放 |
| `can_retrace_after_ik_failure` | 原函数的允许条件；撤离逆放置/搬运路径前，按实测松开夹爪和释放后场景重新检查 |
| `Controller.move_to` | 原函数直接绑定；保留位置积分、原来 120 步路点预算及接近放置终点时的有限延长；输出通过 native action layout 转为实际 composite 动作 |

实现位于 `policy/robocasa_v10.py`，环境映射位于 `adapters/robocasa/v10_preview.py`，反馈控制绑定位于 `control/robocasa_v10_controller.py`。共享参数直接读取冻结 v10 配置，逐项确认在原 130 个任务中一致，不复制某个任务的坐标、轨迹或特殊参数。每回合保存 `LIBERO_V10_RULE_SOURCES.json`，包括原函数的模块和内容指纹。

## 必须由新环境提供的绑定

- PandaOmron 的七个机械臂关节、两个手指、TCP 和 composite controller 动作布局；底盘/躯干仍保持实测初态。
- 原生物体、fixture、支撑表面和成功函数。物体的 `reg_int` 用于容器内部；不能把包含锅柄、带偏心的 `reg_bbox` 当作锅内区域。圆形开口额外检查真实半径。
- 同步 RGB-D 与实例掩码继续产生 GraspGen 的部分点云。仅放置几何改为直接复用原采样函数，采样 4,096 个碰撞表面点，明确标记为 teacher 几何，绝不加入 GraspGen 输入。
- 原接触阶段以真实 MuJoCo 几何为检查依据。移除 r7 对局部接触路径额外增加的球模型否决条件；全局 cuRobo 路径保留球约束和原生几何预检，真实执行保留接触监测。
- 恢复原版的实际关节限位恢复、路径预检、4 mm 中间点/2.5 mm 终点位置精度、1 mm 接触穿透检查、0.8 rad 关节分支偏差保护，以及 0.8 的积分增益、8 mm 积分上限。放置接近目标且持续改善时，按原规则允许最多额外 240 步；仍受本回合总预算约束。观测状态不裁剪，不修改真实 qpos。

RoboCasa 有些原生目标还要求松手后夹爪远离物体，因此原版上抬撤离后可继续沿搬运来路撤回，或用 cuRobo 关节目标撤离；只有经过实测状态预检的路径才能执行。最终仍使用原生成功判定。

## 配对验证

`tools/robocasa/run_pairs.py` 读取 r7 的冻结矩阵，复测指定失败初态。`--replay-grasps-from` 要求任务、seed、split、scene XML、完整初始 qpos 和目标点云一致，才使用之前保存的同一候选池。不匹配会明确失败，不重新抽样后冒称配对。

```bash
python tools/robocasa/run_pairs.py --baseline-root /data/e1-pick-place-r7 \
  --cases PickPlaceCounterToStove:2 PickPlaceCounterToStove:4 \
          PickPlaceCounterToStove:1 PickPlaceSinkToCounter:0 \
  --gpus 0 1 2 3 --paths /data/paths.local.json --output /data/v10-pairs-r8
```

这四个是已知失败开发样本。候选池配对回合不重新运行 GraspGen，不能将模型推理耗时记为正常端到端推理耗时，也不作为未见任务泛化成绩。之后的完整同版矩阵应独立运行并重新调用 GraspGen。

**预算必须显式选择：** 默认 `--budget-profile native` 使用 RoboCasa 原生 horizon。增加 `--budget-profile libero-v10-teacher` 才使用原 LIBERO v10 的 5000 步采集预算；两种模式都保留原生 horizon、实际动作计数和原生目标，并单列 `native_budget_success`。超过原生预算的成功不能作为原生预算内成功；模型、场景、碰撞约束和成功函数不会因采集模式而变更。总墙钟预算仍为 900 秒。机构、导航、复合能力仍未通过厨房物理验收，默认关闭。

普通重新推理采集：

```bash
python robocasa.py run PickPlaceCounterToStove --seed 2 --gpu 0 \
  --paths /data/paths.local.json --budget-profile libero-v10-teacher --output /data/runs/v10-collect-001
```

配置中保留的 r7 `candidate_check_limit`、`transfer_clearance_candidates_m` 不控制 v10 抓放搜索；实际值以 `libero_v10_rules` 和原函数为准。局部 IK 取代全局规划的消融尚未为该继承 selector 实现，配置会显式拒绝，不能把实际 cuRobo 调用记录为局部 IK。

## 本轮检查

80 项 RoboCasa 检查、121 项原 LIBERO 检查均通过。39 个冻结原始文件及 1,040 组 LIBERO 有效配置保真检查通过。

最终 actor 指纹为 `387084f9bee3cb24489e3cfa7c33815aab7a37dfb6408af09002bde3c0687a67`。同版物理记录如下，**三组不合并成功率**：

| 条件 | 任务/初态 | 完整结果 | 原生 600 步内完整成功 |
| --- | --- | --- | --- |
| 原生预算、复用 r7 候选池 | CounterToStove seed 1、2、4 | 0/3；全部失败视频保留 | 0/3 |
| 显式 5000 步采集、相同候选池 | CounterToStove seed 2 | 成功，730 步，178.60 秒 | 否 |
| 显式 5000 步采集、重新运行 GraspGen | CounterToStove seed 2 | 成功，710 步，229.38 秒 | 否 |

seed 1 的真实抬升保持检查没有保持双侧抓取，物体抬升约 0.000063 m；后续退离也未通过几何检查。seed 2 在释放后撤离时到达原生 600 步；同候选池采集模式完成了任务。seed 4 在真实搬运中发生物体与台面约 5 mm 的接触；物体相对末端的位姿在接触附近明显变化，尚不能仅凭终态确定是滑动先发生还是接触导致位移。

五个视频均已解码验证，N 个动作/N+1 个观测对齐有效，运行期间 actor 指纹一致。额外成功回合导出 4 个 teacher 关键点，仍待学生观测充分性审核。新版本未完整重跑十回合，也未完成未见任务/其他操作验收。可核对的摘要见[验证记录](../../migration/robocasa/VALIDATION_V10_MIGRATION_20261112.json)。
