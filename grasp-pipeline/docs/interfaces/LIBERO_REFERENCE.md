# LIBERO 实现与接口参考

本参考由第二阶段源码和原有 v10 记录生成。它不启动环境、不运行抓取，也不把旧驱动注册成新的通用适配器。

## 现有功能对应关系

| 接口 | 当前实现 | 需要补的绑定或转换 |
| --- | --- | --- |
| 环境 | `EnvAdapter.reset/step/close` | 新的回合/状态 ID；把混合 done 的四元组转换为明确终止原因 |
| 任务 | `run_task.py` 读取目标，`task_policy.schedule` 编排，原生 `_eval_predicate/check_success` | TaskSpec 和原生判定封装；独立补充语言顺序约束 |
| 机器人 | `EnvAdapter.state`、Panda MJCF、`frames.grasp_to_control` | 发布模型、关节和动作说明；从机器人状态中分离特权物体位姿 |
| 场景 | `export_scene`、`regions`、`placement_geometry`、`articulation` | 场景版本、可解析资源、区域和关节数据约定 |
| 规划 | `BridgePlanner.plan_pose`、`plan_cartesian` | 规范请求/结果、后端和起点引用；局部 IK 仍需要 MuJoCo |
| 预检 | `ContactMonitor.validate_path` 与局部 IK 的 snapshot/restore | 对新引擎实现相同的查询和恢复保证 |
| 执行 | `Controller.follow_path/move_to/hold` | 标准执行报告与原生动作、终止、夹爪映射 |
| 记录 | `Recorder.capture/save`、旧打包的 KEYPOINTS | 保存前后原始观测、准确采集序号和真实决策引用 |
| 能力 | 现有函数/配置和新增离线 readiness | 尚无运行时注册或自动门控；当前 binding 标记为 reference_mapping_only |

精确源文件、函数、行号、哈希与缺口见 [LIBERO_V10_BINDINGS.json](../../contracts/LIBERO_V10_BINDINGS.json)。11 个源码锚点覆盖九类接口，校验工具会检查锚点源文件未变动。

## 真实数据样例

[libero_v10_reference.json](../../contracts/examples/libero_v10_reference.json) 来自原 v10 的“橙汁盒放篮子”记录，包含真实初始机器人状态、2048 点目标点云、原始目标、42 个碰撞几何、22 个阶段关键点及动作/状态计数。模型关节限位从保存的 merged MJCF 提取；点云直接读取原 NPZ，没有重新推理。

样例保留这些缺失：

- 原记录未保留逐帧原始 RGB-D、实例掩码和完整相机标定数组。
- 点云元数据未记录准确采集状态序号，不把它假设为状态 0。
- 该样例不另行恢复目标区域几何；关节样例也未伪造。
- XML 未显式写出重力，导出器没有把默认值冒充成记录值。
- 此样例的 42 个几何均为导出的盒体；若其他场景包含 mesh，原服务器 URI 需要另行解析，不能直接标为本地可用。机器人模型自身的资产依赖也需部署方准备。
- 原语言指令顺序约束和训练质量没有做完整认证。

因此该样例的结构校验可以通过，execution_ready 应为 false。它用于展示接口数据和明确缺口，不是可拷贝运行的完整场景包。已有在线 LIBERO 管线能在运行时查询许多缺失项；离线交付没有这些字段，不表示原实验没有使用它们。

## 成功口径不变

[libero_v10_outcomes.json](../../contracts/examples/libero_v10_outcomes.json) 是全部 130 条旧 RESULT 的规范化表示。校验要求保持：官方最终成功 88、完整管线成功 86、排除已知顺序错误后的候选成功 85。没有新增任务执行或混入历史版本。

规范化记录把原原因保留在 failure.details，无法可靠重分类的原因写 unknown。两条官方 true 但管线未完整结束的记录仍保持这个组合；已知堆叠顺序错误仍单独标记。

这些旧记录没有完整观测和训练审核，严格的 sft_eligible 全为 false。它不把 85 条候选成功变成新的任务失败，只说明还不能按本接口宣称训练数据已完整验收。

## 下一阶段边界

第四阶段已提供[外部维护者的实施指南](../integration/README.md)，包含文件修改点、配置和调试验收。当前还未让旧驱动经由新 Protocol 工作，也未消除全部直接访问 MuJoCo/LIBERO 的调用；不能把文档完成表述为已接通其他环境。
