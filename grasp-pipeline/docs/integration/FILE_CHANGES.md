# 需要新增和修改哪些文件

下列现有路径均相对本代码根目录。**“拟新增”是接入方的工作建议，仓库尚未提供对应实现。** 当前运行链尚未调用第三阶段 Protocol，因此新增接口类只是起点。

机读定位见 [FILE_CHANGE_MAP.json](FILE_CHANGE_MAP.json)，包含现有函数行号和源码指纹；它不是自动改写代码的脚本。

## 接入方拟新增的文件

| 拟新增位置 | 内容 | 接入完成前必须成立 |
| --- | --- | --- |
| `src/graspgen_libero_bridge/adapters/<bench>/` | 环境/任务/机器人/场景/执行/记录提供方，可按需要拆文件 | 实现所选九类方法职责，原生成功和终止语义明确；不能只转发 reset/step |
| `apps/<bench>/run_task.py` | 本环境的回合和阶段编排 | 连接提供方、策略、模型、规划、预检、执行与记录，阶段不 reset |
| 本环境的 planner/preview/executor 后端 | 机器人模型、局部 IK、引擎预检、实际动作映射 | 与所选硬件和引擎一致，并声明实际后端 |
| 新 task/config/profile 文件 | 自己的任务目录、部署信息、能力要求与参数版本 | 不向冻结 LIBERO 清单塞入异构任务，不冒用 LIBERO teacher profile |
| `tests/<bench>/` | 坐标、动作、生命周期、原生结果、回放和接口测试 | 既有回归通过，本环境的真实任务验收单独留证 |

可以直接使用 benchmark 原生 API，也可包装已有 RPent/RLinf 等环境客户端；不要求通过 RPent。提供方可以合并实现多个 Protocol，不必部署九个服务。

## 现有代码必须逐项处理的地方

| 现有位置 | 当前耦合点 | 接入方需要修改、包装或替代什么 |
| --- | --- | --- |
| [runtime/cli.py](../../src/graspgen_libero_bridge/runtime/cli.py) | 完整任务固定启动 `apps/libero/run_task.py` | 另建显式入口，或实现并测试新的 dispatch；当前没有可直接启用的插件注册函数 |
| [runtime/config.py](../../src/graspgen_libero_bridge/runtime/config.py) | 读取固定 LIBERO 清单、15 个覆盖值和六路径格式 | 新增独立配置加载，保持 benchmark/robot/action 参数与部署路径分开 |
| [apps/libero/run_task.py](../../apps/libero/run_task.py) | 原始 BDDL、私有目标查询、环境共享、阶段构造 | 在新 driver 中接入 TaskPort 与本环境提供方；原任务/初态不可被阶段初始化覆盖 |
| [policy/evaluate.py](../../src/graspgen_libero_bridge/policy/evaluate.py) | `PickPlace` 构造固定 `EnvAdapter`，内部也读 task_env、sim 和区域 | 拆出或注入环境依赖，或写本环境编排复用纯计算部分；不能只把 adapter 属性换成另一个 env |
| [adapters/libero/env_adapter.py](../../src/graspgen_libero_bridge/adapters/libero/env_adapter.py) | 单 Panda、OSC、官方初态、私有索引、混合 done | 新提供方实现对应状态/动作/初始化，保留原生 terminated、truncated 和 success |
| [perception/pointcloud.py](../../src/graspgen_libero_bridge/perception/pointcloud.py) | `robosuite_target_points` 与完整 MuJoCo 几何入口依赖原环境 | 复用通用采样/反投影；本环境负责掩码、深度米制化、标定、目标 ID 和来源 |
| [adapters/libero/scene.py](../../src/graspgen_libero_bridge/adapters/libero/scene.py) | 从编译 MuJoCo 接触几何导出世界 | 为新环境导出相同语义的场景，解析资产、缩放、附件、未知空间与更新版本 |
| [regions.py](../../src/graspgen_libero_bridge/adapters/libero/regions.py)、[placement_geometry.py](../../src/graspgen_libero_bridge/adapters/libero/placement_geometry.py)、[articulation.py](../../src/graspgen_libero_bridge/adapters/libero/articulation.py) | 原目标区域、物体属性、site、关节和把手查询 | 提供本环境的区域/支撑/关节语义，保持局部轴向和原生目标阈值 |
| [planning/planning_geometry.py](../../src/graspgen_libero_bridge/planning/planning_geometry.py) | `PandaMJCF`、固定 arm/finger 名称和碰撞球 | 同模型也需 FK/碰撞对齐；不同机器人提供新的模型和配置，不只改机器人名字 |
| [planning/planner_worker.py](../../src/graspgen_libero_bridge/planning/planner_worker.py) | 构造 Panda 模型，q_arm 为七维，持物与场景格式固定 | 接入所选模型/后端、状态与 goal 变换，绑定计划版本，保留原始规划失败 |
| [planning/cartesian.py](../../src/graspgen_libero_bridge/planning/cartesian.py)、[planning/grasp_contact.py](../../src/graspgen_libero_bridge/planning/grasp_contact.py) | MuJoCo Jacobian、接触及临时物理状态 | 同引擎映射索引与模型；不同引擎实现等价后端或明确缺失，不可直接跳过局部阶段 |
| [control/controller.py](../../src/graspgen_libero_bridge/control/controller.py) | OSC、机器人/夹爪私有字段、接触检查、记录器 | 替代/封装跟踪、限位、预检、开合及结果；完整保留六自由度目标朝向 |
| [policy/skills.py](../../src/graspgen_libero_bridge/policy/skills.py)、[policy/surface_packing.py](../../src/graspgen_libero_bridge/policy/surface_packing.py) | 直接访问 sim、fixtures、关节与目标区域 | 把数据查询和实际接触执行接到本环境，不能仅复用技能名字 |
| [policy/task_policy.py](../../src/graspgen_libero_bridge/policy/task_policy.py) | all 目标、指定谓词和英文 push 路由；排序不能覆盖所有语言顺序 | 只在目标语义确实相同且顺序已验证时复用；否则新增任务编译/技能逻辑并记录新策略版本 |
| [models/graspgen_worker.py](../../src/graspgen_libero_bridge/models/graspgen_worker.py) | 官方 Panda 夹爪与 2048 点配置检查 | 同夹爪可保留；不同夹爪需匹配模型/权重及标定，不要删掉检查假装兼容 |
| [contracts/validation.py](../../src/graspgen_libero_bridge/contracts/validation.py) | `readiness` 固定 LIBERO/Panda/teacher，包含原任务路由 | 通用 `validate_payload` 可用；新环境需要对应能力检查/版本配置，不应覆盖旧 profile 骗过检查 |
| [common/layout.py](../../src/graspgen_libero_bridge/common/layout.py) 与启动/RPC 调用处 | checkout 路径、模块名和工作进程查找 | 确认 worker 使用新模块和解释器；公共 RPC、坐标、错误工具按原语义复用 |
| [tools/legacy_evaluation](../../tools/legacy_evaluation/README.md) | 原服务器和 `versions/v10` 目录布局 | 为新环境写独立评测/汇总，不直接套用旧队列来计分 |

`placement.py`、`transfer.py`、`frames.py`、`grasp_candidate.py` 等纯计算可以优先复用，但输入坐标、尺寸、重力和语义必须一致。以上是改动定位，不是承诺每一处都要重写；保留的部分也要验证实际调用方。

## 建议的接线顺序

1. 先让新提供方产生 Observation、RobotSpec、TaskSpec、SceneSnapshot，验证数据而不执行动作。
2. 在新 driver 里拿到同一回合的任务、状态和场景，运行对应能力检查。
3. 把原策略中的环境查询替换为这些提供方的数据；保留已决定复用的计算规则和参数，任务语义变化单独版本化。
4. 调通 GraspGen 和所选规划 worker，再接局部预检与执行；对每个阶段记录实际后端。
5. 接上真实原生成功、终止和 RecordingPort，最后才注册新的 CLI 路由或供外部 runner 调用。

示意流程是“观测/任务 → 阶段目标 → 规划 → 预检 → 执行 → 新观测/原生判定 → 记录”，目前没有能直接替你运行这一新环境流程的公共 runner。

当前 driver 有部分全局运行状态；不要通过 import `apps/libero/run_task.py` 来当库调用，它会解析命令并执行。新入口应显式组织自己的生命周期。

## 防止改动混入基线

冻结参考保留不动，新适配在独立分支/目录实现。现有 `verify_preservation.py` 会拒绝未记录的原代码变化；若为了接入而修改共享实现，要建立新版本并说明变化，不能把校验关掉后仍称与 v10 等价。单独的接口定义和文档也不能当作真实接入成绩。
