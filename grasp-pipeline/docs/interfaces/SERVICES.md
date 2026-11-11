# 方法职责与运行约定

## 九类接口

一个适配对象可实现多个接口，不要求创建九个进程。`Payload` 是对应 Schema 对象；Python 签名见 `contracts/ports.py`。

| 接口 | 方法和输出 | 责任方与约束 |
| --- | --- | --- |
| `EnvironmentPort` | reset/observe → Observation；step → StepResult；close | 接入方创建和推进环境，维护 seed、初态、回合和状态序号 |
| `TaskPort` | describe_task → TaskSpec；check_native_success/check_subgoal → bool 或 null | 接入方解释本环境原任务；判定原生成功，不改原目标来配合策略 |
| `RobotPort` | describe_robot → RobotSpec；robot_state → RobotState | 接入方提供模型、命名关节、基座/TCP、实际状态与控制约定 |
| `ScenePort` | scene_snapshot → SceneSnapshot | 接入方提供碰撞场景、相关对象、区域和关节几何，标记更新和未知区域 |
| `PlanningPort` | plan(request, robot, observation, scene) → PlanResult | 规划后端接收策略给出的数值目标；声明实际使用 cuRobo、局部 IK 或其他后端 |
| `PreviewPort` | validate_and_restore(plan, contact_policy) → PreviewResult | 接入方或同引擎后端负责真实几何预检与无副作用恢复 |
| `ExecutionPort` | execute(request, plan) → ExecutionReport | 接入方将路径变成原生动作并闭环执行，落实夹爪语义和终止处理 |
| `RecordingPort` | record_step(before, action, after, phase, decision_record)；finalize → TrajectoryRecord | 每个真实动作记前后观测，保留失败与实际决策来源 |
| `CapabilityPort` | capabilities → Capabilities；check_requirements → ReadinessReport | 执行前列出支持能力和缺失条件，不能把未实现技能算为尝试失败 |

当前代码已具备许多对应功能，但它们不是这些 Protocol 的现成实现。LIBERO 的具体函数与缺口见 [参考映射](LIBERO_REFERENCE.md)。

## 生命周期

1. 读取原任务和机器人说明，确认 profile、观测来源、必要几何及原生成功接口。
2. `reset(task_id, seed, initial_state_id, episode_id)` 建立一个回合；必要的初态稳定过程显式记录，之后才开始动作序号。阶段之间保持同一回合。
3. `observe` 和状态、场景查询不推进物理；同一个序号的机器人、场景、目标观测应描述同一时刻。没有新观测时保留旧采集序号。
4. 策略生成阶段目标，GraspGen 在抓取阶段生成候选；规划器接收 `PlanRequest`。计划失败在 planning 分类结束或按原策略继续搜索，不记作夹取失败。
5. 执行前确认起点和场景未过期，预检通过且状态已恢复；有真实运动后不能再按“完全没执行”的候选状态重试。
6. `step` 每次只推进声明的控制周期。原生 terminated、truncated 和 native_success 分开记录；原生成功不自动等于环境已终止，已终止也不能继续发送动作。
7. 到位后根据阶段闭合或打开夹爪，结合真实接触、抬升、释放和原生目标检查。闭合命令成功不能代替抓稳证据，打开命令成功不能代替稳定放置。
8. 回合结束保留最后状态、全部已执行动作、阶段、原因和视频；部分失败也可以产出有效的不完整记录。

## 不同技能额外需要什么

| 技能 | 共同能力之外的输入 | 结束依据 |
| --- | --- | --- |
| 抓放 | 可用目标点云、目标/容器几何、放置区域、持物相对变换、支撑接触许可 | 执行释放/撤离后查询原任务，单独记录抓取证据 |
| 推移 | 物体与支撑面、接触点和推移约束、目标区域、持续接触检查 | 原目标谓词及执行反馈 |
| 门/抽屉/旋钮 | 关节类型、轴、锚点、限位、当前 q、把手几何、原生开关规则 | 关节状态与原生 open/close/turnon/turnoff 判定 |
| 复合操作 | 每个子目标、依赖和语言顺序约束、阶段间状态与持物变化 | 全部原生目标和指令顺序分别核对 |

GraspGen 的当前 Panda 模型不自动生成门、抽屉、旋钮的操作计划。cuRobo 能规划给定约束下的运动，但需要上层提供接触/关节技能与目标序列。未知技能或谓词应返回 unsupported，不能仅把相同抓放顺序套上去。

能力检查按技能增加要求：抓放需要目标点云，关节操作需要轴、锚点与把手，不能要求开抽屉也必须先生成 GraspGen 点云。当前 profile 仍沿用旧策略的 all 目标语义和英文 push 关键词路由；其他逻辑或语义需要明确转换，不能默默更换任务含义。

## 错误分类

| 情况 | 接口处理 |
| --- | --- |
| 缺深度/目标点云/有效标定 | perception 或 readiness blocker；重新观测或明确不支持 |
| 缺机器人模型、动作映射、网格资源 | configuration/integration；禁止漏掉几何后继续假定可执行 |
| 缺少原生目标语义或未知谓词 | task/unsupported；保留任务定义 |
| 起点限位或碰撞、预算内无路径 | planning 的 invalid_start/no_path/budget_exhausted，保留起点和诊断 |
| 真实几何预检拒绝 | preflight；不写 grasp_failed |
| 预演后状态未恢复 | integration，保存失败并中止该回合 |
| 实际执行碰撞、跟踪超时 | control；保留已执行动作数和最终状态 |
| 执行夹爪后抬升不达标 | grasp_check，需要真实执行证据 |
| 放置后原目标不满足 | goal_check；与原生 true 但管线未结束分别记录 |
| 步数或墙钟预算结束 | budget，区分控制步数、规划请求预算和整回合墙钟预算 |

`Failure` 同时带 domain、code、phase、message、details。未知原因保留 unknown 与原日志，不根据视频长度直接推断模型能力。

## 两条必须保留的边界

预演应使用模型副本，或完整保存恢复积分状态、控制输入、暖启动、外力等可能修改的状态；如会触及控制器或随机状态，也必须恢复。当前局部规划不推进真实物理时间，换引擎不能通过瞬移物体来执行规划。

路径规划与真实执行各声明实际后端。若把 cuRobo 终点传给环境 `move_to` 让它再次规划，实际执行路径不能仍全部标成 cuRobo；必须记录重规划和控制方式。
