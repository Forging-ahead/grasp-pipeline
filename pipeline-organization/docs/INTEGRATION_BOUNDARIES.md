# 外部 benchmark 接入边界盘点

本文件保留第一阶段的依赖盘点。第三阶段已将这些要求明确为 [第一版接入约定](../../grasp-pipeline/docs/interfaces/README.md)，包括字段、方法、校验和 LIBERO 参考；当前运行链尚未按统一接口重接，不能理解为已有通用适配器。

当前迁移要求见[从 libero 分支继承实现](../../grasp-pipeline/docs/integration/BASELINE_FIRST.md)：新 benchmark 必须基于 `libero` 分支继续开发，优先直接复用已有策略、规划、接触预检与控制规则，补齐环境适配，不从零重写简化管线。

## 我们整理什么和接入方实现什么

我们负责整理自己的候选生成、阶段目标计算、规划调用、轨迹执行编排和记录格式，给出 LIBERO 的参考实现与清楚的边界。外部 benchmark 接入方负责实现其环境接口，提供机器人和场景模型，把规划输出转成可执行动作，并调用自己环境的任务成功判定。

更换 benchmark 和更换机器人是两项工作。目前实际限定单 Panda、Panda 平行夹爪、七个机械臂关节和 `OSC_POSE`。即使两个 benchmark 都用 MuJoCo，也不能据此假定对象命名、任务 API、控制模式和机器人模型相同。

## 接入方必须提供的能力

| 能力 | 当前 LIBERO 实现 | 外部接入需要做什么 | 主要代码位置 |
| --- | --- | --- | --- |
| 环境生命周期 | suite、任务名、官方初态，单回合连续执行 | 创建、reset、step、close，保留实际 seed、初态、步数、terminated/truncated；阶段之间不能重置 | `env_adapter.py`、`run_task.py` |
| 相机与目标点云 | 深度、相机内外参、仿真实例掩码，世界坐标 XYZ | 提供相同含义的米制点云，或深度与校准及目标掩码；明确来自传感器还是仿真真值 | `pointcloud.py`、`evaluate.get_points` |
| 任务目标和技能计划 | 读取 BDDL 目标、区域和对象归属，规则排序 | 将原任务转为显式目标与前置条件，定义顺序；不要求别的环境使用 BDDL，不能只改任务名字 | `task_policy.py`、`run_task.py` |
| 机器人状态 | 七关节、手指状态、速度、基座和末端 site 位姿 | 提供命名关节及顺序、单位、限位、实际测量状态、基座和 TCP 变换 | `env_adapter.state`、`frames.py` |
| 机器人和夹爪模型 | 从 robosuite Panda MJCF 生成 cuRobo 模型 | 提供相应运动学与碰撞模型，校验 FK；换夹爪要验证 GraspGen 权重适用性和 TCP 偏移 | `planning_geometry.py`、`planner_worker.py`、`graspgen_worker.py` |
| 场景碰撞模型 | 直接导出 MuJoCo 接触几何和物体位姿 | 提供障碍物几何、位姿、附着物更新和接触许可；标明感知未知区域及更新方式 | `scene.py`、`placement_geometry.py` |
| 局部规划和碰撞预检 | MuJoCo Jacobian、真实接触查询、临时状态保存恢复 | 同引擎也需映射模型及索引；不同引擎需实现对应 IK、路径检查或替代后端，不能只导出障碍盒 | `cartesian.py`、`controller.ContactMonitor`、`grasp_contact.py` |
| 运动执行与夹爪 | 规划路点转换为七维 OSC 增量动作 | 按环境动作模式实现姿态和夹爪命令、闭环反馈、超时、限位及碰撞处理，保留抓取姿态方向 | `controller.py`、`frames.py` |
| 放置区域与关节技能 | 目标区域边界、支撑面、物体几何、把手、关节轴/锚点/区间 | 映射相应语义和几何；对缺失信息显式报不支持或由感知补足 | `regions.py`、`placement.py`、`articulation.py`、`skills.py` |
| 任务成功与记录 | 原始官方目标判定、阶段检查、视频、状态/动作/阶段 | 调用该 benchmark 原生成功判定，分别记录管线完成和任务成功，保存全部失败及观测来源 | `run_task.py`、`controller.Recorder`、测评报告工具 |

## 不能只改一个环境入口的原因

当前环境访问并未集中在 `EnvAdapter`。`controller.py`、`cartesian.py`、`scene.py`、`regions.py`、`placement_geometry.py`、`grasp_contact.py`、`skills.py` 和 `articulation.py` 等直接读取仿真模型、接触、关节或任务注册表。详细行号可在 [SOURCE_INVENTORY.json](../audit/SOURCE_INVENTORY.json) 的 `coupling_evidence` 中查找；该搜索包含注释命中，需结合函数正文解读。

后续需要先把这些访问归到环境、机器人、几何和执行的边界，再写接入示例。当前没有一个可以承诺“只实现 reset/step 就接入成功”的接口。

## 数据约定需要固定

- 内部主位姿使用 `world_T_*` 的 4×4 变换矩阵，平移为米；必须说明源坐标系、目标坐标系与变换方向。
- 区分 GraspGen 的夹爪参考点、实际控制 TCP、机器人基座和相机；0.097 m 是当前 Panda 适配值，不能给所有机器人照搬。
- 对每个四元数接口明确 `wxyz` 或 `xyzw`；不能混用。关节角使用弧度，手指位置使用该模型约定。
- 明确单臂、双臂、移动底盘的关节和动作顺序。不要截断多余维度来假装适配。
- 路径由哪个后端负责要明确：如果向环境原生 `move_to` 再提交目标让其重新规划，须记录实际规划器，不能仍把执行路径全部归为 cuRobo。
- 真值教师模式与标准视觉模式分开配置和记录。目标点云可见不等于已获得完整环境碰撞模型；未提供的信息不能静默从另一接口泄漏进来。

## 接入文档后续验收顺序

1. 核对环境 reset、初态、相机与原生任务成功判定，保存一个静态场景记录。
2. 对齐关节 FK、基座、TCP、相机和目标点云，检查单位、轴向和夹爪开合符号。
3. 导出场景，核对碰撞几何、持物附着和阶段接触许可。
4. 测试一个自由空间位姿移动及夹爪动作，再检查预检和状态恢复是否改变真实环境。
5. 在独立回合尝试单次抓取、抬升、放置和释放，按原生成功判定验收。
6. 再接入该 benchmark 支持的复合或关节技能，验证原始目标与指令顺序。
7. 统一保存成功、失败、超时、未支持和技术错误；外部 benchmark 单独出成绩，不能沿用 LIBERO 的 86/130。

RoboTwin、RoboCasa 的具体版本、机器人、控制器和评分入口由接入方选定；本轮后续文档可以基于已有代码说明差异，但不会替它们实现适配或宣称已有测评成绩。
