# RoboCasa 分支代码与改动清单

本分支从 `libero` 基线继承成熟实现，为 RoboCasa 增加环境绑定与必要后端；旧 LIBERO actor、配置和冻结 v10 保留。统一入口见[分支说明](README.md)，实际继承的原函数见[v10 迁移记录](V10_MIGRATION.md)。下列路径相对 `grasp-pipeline/`；代码存在不等于通过物理验收。

| 新增路径 | 用途 |
| --- | --- |
| `robocasa.py`、`apps/robocasa/` | 独立配置、环境检查、控制验证和单任务入口 |
| `configs/robocasa/` | 12 个内置语义任务绑定、r13 共享策略与独立机器路径示例；部分保留的 r7 字段不控制 v10 搜索，不存逐 seed 世界坐标 |
| `adapters/robocasa/`（在 `src/graspgen_libero_bridge/` 下） | 原生环境生命周期、动作切片、实例掩码与 RGB-D、场景/目标区域/机构查询、记录与失败类型 |
| `planning/robocasa_*.py` | PandaOmron 的 Panda 模型校验；cuRobo worker 包装、持物与双模型预检；局部 IK、受限路径简化和导航几何 |
| `control/robocasa_executor.py`、`robocasa_v10_controller.py` | 将原 v10 反馈控制绑定到原生动作布局；限位、动作预算、接触监测与原生目标稳定结束 |
| `policy/robocasa_*.py` | 抓放阶段编排，以及默认关闭的机构/移动原型；阶段之间不 reset |
| `runtime/robocasa_*.py` | 配置启动、连续回合、内容指纹、worker 缓存隔离、总墙钟预算和进程清理 |
| `tools/robocasa/` | 资产、诊断、矩阵、配对、65 项覆盖、报告与审核工具；详见[工具索引](../../tools/robocasa/README.md) |
| `tests/robocasa/` | 104 项 RoboCasa 检查（加上原 LIBERO 的 121 项，共 225 项通过）；不能替代真实厨房任务的稳定性验证 |
| `deploy/robocasa/` | 与 LIBERO 分离的仿真环境及部署说明 |
| `docs/robocasa/`、`migration/robocasa/` | 计划、实验设计、实际状态及可核对的验证摘要 |

保留并实际调用原有 GraspGen worker、RPC、坐标变换、部分点云处理与规划几何；没有重新训练生成器/判别器。cuRobo 包装复用原 worker，但用新 robosuite 资产建立和校验 Panda 模型。完整场景几何用于 teacher 规划与检查，GraspGen 的输入仍为目标可见部分点云。

公开九类 Protocol/Schema 尚未完成绑定。当前新 driver 调用本轮内部提供方 API；它已经产生真实物理动作，但仍是一个具体环境实现，不能当作完成了通用接入框架。

本轮未开放机构、导航、复合及多体持物能力的默认开关。它们只有代码原型和几何/绑定检查，尚无厨房物理任务验收结果。没有修改共享旧入口来自动切换 benchmark。
