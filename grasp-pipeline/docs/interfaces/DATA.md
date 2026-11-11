# 数据字段和坐标约定

每类公开数据包都含 `schema_version="1.0"` 和对应的 `payload_type`。未记录的信息使用 Schema 允许的 `null` 并解释缺失原因，不能用零、单位矩阵或空场景冒充有效数据。完整字段表在 Schema 的 `$defs` 中。

## 统一坐标与时间

- 平移、点云、几何尺寸和移动关节使用米；旋转关节使用弧度；时间使用秒。
- `A_T_B` 把 B 坐标系中的列向量变换到 A：`p_A = A_T_B @ p_B`。所有位姿为 4×4 齐次矩阵，旋转正交且行列式为 +1，最后一行为 `[0,0,0,1]`。
- 世界坐标以环境公开的 `world` 为准。相机采用 OpenCV 轴向：X 向右、Y 向下、Z 向前；原 OpenGL 深度和掩码需要一起转换，禁止再重复翻转外参。
- 接口不传含糊的四元数数组。旧 cuRobo/scene 七维 pose 为 `[x,y,z,qw,qx,qy,qz]`，SciPy 默认返回 xyzw；适配时显式转换为矩阵。
- `episode_id` 标识同一物理回合，`state_index` 是执行动作后的状态序号。`reset` 后的可执行初态为 0，动作 i 把状态 i 变为 i+1。规划和预演不增加它。
- `sim_time_s` 不等于墙钟耗时。点云、场景和机器人状态各自携带采集序号，不允许把旧点云标成最新观测。

## 来源与文件

`Source.types` 可同时包含 sensor、robot_state、configuration、task_definition、simulator_truth、estimation、memory、code_policy；`evidence` 指向原始记录或调用来源，`description` 说明转换。推导结果继承输入的来源：用真实深度加仿真掩码生成的点云同时标记 sensor 和 simulator_truth。

`Artifact` 包含 URI、媒体类型、SHA256、可用性和说明。可用性为 local、external 或 missing。网格顶点摘要不能冒充 OBJ 文件摘要。URI 和 local 都相对于提供方的解析环境，跨机器后必须重新解析和核验；字段校验不证明资源文件已存在。

## 观测和机器人

| 数据 | 主要字段 | 必须明确的语义 |
| --- | --- | --- |
| `Observation` | episode/observation ID、state index、sim time、mode、robot_state、cameras、target_cloud | teacher 或 sensor 显式区分；相机数据与特权场景分开 |
| `RobotState` | joint_names/q/qvel、finger_joint_names/finger_q、world_T_base、world_T_tcp、source | 名称和数组顺序相同；保留测量值，不裁剪真实关节状态来满足规划器 |
| `Camera` | rgb/depth_m/instance_mask 文件、intrinsic、world_T_camera、capture_state_index | 深度已米制化；掩码与图像尺寸、像素方向一致，明确掩码来源 |
| `Cloud` | target_entity_id、points[N,3]、sample_count、frame=world、units=m、partial、capture_state_index | 对应明确目标，点值有限；不能把局部坐标或厘米直接当世界米制点云 |
| `RobotSpec` | robot_model、model_ref、命名关节及限位、base/tcp frame、gripper_model、grasp_T_tcp、动作编码/维度/频率 | 机器人模型与实时状态属于不同输入；换机器人需要匹配运动学、碰撞和控制实现 |

GraspGen 当前 worker 要求至少 16 个有效目标点，再采样为 2048 点，使用 XYZ；RGB 不直接进入该模型。缺少可见点时，接入方可以重新观测或提供有来源的记忆/重建，不得直接把不可见物体位置视为已知。

只有单目 RGB 和机器人状态不能直接满足当前抓放 profile；还需要接入方提供米制目标点云、相关几何和目标定位的可靠来源。接口没有自动实现单目深度、目标检测或场景重建。

当前 Panda 的 `grasp_T_tcp` 是沿 GraspGen 原生 +Z 偏移 0.097 m，得到 `world_T_tcp = world_T_grasp @ grasp_T_tcp`。它来自原配置，不是适用于所有机器人的常量。cuRobo 后端再使用 `base_T_world @ world_T_tcp`，不要在调用方和后端重复转换。

## 任务和场景

| 数据 | 主要字段 | 语义 |
| --- | --- | --- |
| `TaskSpec` | benchmark/task ID、language、definition_ref、entity/region IDs、goal_logic、goals、required_order、instruction_constraints_complete | 不要求其他 benchmark 使用 BDDL；保留原任务成功条件与额外顺序要求 |
| `Goal` | goal_id、predicate、arguments、source | 对象和区域引用必须存在；未知谓词不能静默丢弃 |
| `SceneSnapshot` | scene ID、state index、gravity、coverage、unknown_space_policy、geometries、entity_poses、regions、articulations | 与计划起点同步；未观测空间不能默认为空，必须声明处理方式 |
| `Geometry` | geometry_id、owner、world_T_geometry、shape、parameters、representation、source | box 尺寸为完整边长；sphere 半径、cylinder 半径和全长、mesh 资源与缩放明确，缩放只用一次 |
| `Region` | region ID、owner、owner_T_region、bounds_min/max、source | 边界属于 region 局部坐标；抽屉/柜体的竖直方向不能默认是局部 Z |
| `Articulation` | owner、joint name/type/q/limits、world axis/anchor、handle geometry IDs、native goal rule | 拉动/转动需要关节几何与原生开关判定，单独一个末端终点不够 |

目标物体完整碰撞几何、目标容器位姿、支撑面和关节状态可来自仿真、感知或记忆，但来源必须留下。规划器不知道未提供的物体；场景接口负责更新它看到的世界，不由 cuRobo 推断任务语义。

持物规划使用测量得到的 `tcp_T_object`，所需末端目标为 `world_T_object_goal @ inverse(tcp_T_object)`。它不是物体回合初态与夹爪之间固定不变的假设；滑动或重抓后要重新测量。

## 规划、执行与预演

| 数据 | 字段重点 | 约束 |
| --- | --- | --- |
| `PlanRequest` | request/episode/scene/robot ID、起点序号、phase、backend、goal、contact_policy、timeout | goal 为 TCP 位姿或明确命名的关节目标，下一阶段由策略给出 |
| `ContactPolicy` | optimizer_excluded_geometry_ids、allowed_contact_pairs、attachments | 优化器排除项和真实预检许可分开；附件逐物体给出 TCP 变换和碰撞几何 |
| `PlanResult` | 输入引用、plan ID、实际 backend、status、waypoints、wall time、failure | planned 才能提供可执行路径；失败结果不得携带伪造的成功路径 |
| `ExecuteRequest` | plan ID、expected_state_index、phase、gripper_command、max_control_steps | 夹爪语义为 open/close/hold，适配方映射到原生符号、宽度或力控制 |
| `ExecutionReport` | start/end index、actions_executed、final state、terminated/truncated、grasp_assessment、failure | 到位和抓稳分别记录；前后状态差应等于实际执行步数 |
| `PreviewResult` | passed、state_restored、前后状态序号、failure | 失败也可记录；恢复失败必须终止该回合，不能继续真实执行 |

`validate_plan_context` 校验计划与当前机器人、场景和回合引用，以及碰撞许可引用的几何 ID。后端仍需做 FK、IK、碰撞检查、路径起点偏差和真实反馈检查；接口结构正确不提供无碰撞或抓取成功保证。

## 成功与训练记录

`TaskOutcome` 单独记录管线是否执行完整、原生任务是否成功、动作数、指令顺序、轨迹质量、观测记录完整性和失败。原生成功为 null 表示未获得判定，不能算成功。

本仓库离线统计约定：完整成功需要 pipeline_complete、native_success 和 steps>0；候选成功进一步排除已知顺序错误。训练可用还要求顺序已确认、质量已接受且观测记录完整。这些是数据审核字段，不会改动现有任务驱动的退出规则。

`TrajectoryRecord` 记录 N 个动作、N 个动作阶段和 N+1 个状态。关键点的 action_index=i 对应 state_index=i+1 的实测位姿；前置观测应关联状态 i。code policy 阶段标签不应改称 VLM 决策，VLM 标签必须附真实决策记录。视频结束停帧不增加动作或状态数。
