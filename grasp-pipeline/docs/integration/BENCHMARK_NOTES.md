# RoboTwin 与 RoboCasa 接入差异

核对日期：2026-11-12。以下区分官方资料、本地 RPent 示例和我们据此提出的接入工作，未进行两个 benchmark 的运行验证。正式接入时仍需固定所选仓库、包装层、任务资产、机器人和配置版本。

来源与本地源码指纹见 [SOURCE_REFERENCES.json](SOURCE_REFERENCES.json)。官方网页和 main 分支只是本次查阅来源，不等于接入方已安装的版本。

## RoboTwin

官方实现使用 SAPIEN；其机器人控制支持 qpos 与 ee 模式。qpos 维数随具体机械臂配置变化，动作包含左右臂和各自夹爪；ee 使用两侧的位姿和夹爪。不能把一个示例机器人的维数当作整个 benchmark 的固定规格。[官方 Base_Task 源码](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/envs/_base_task.py)、[官方控制说明](https://robotwin-platform.github.io/doc/usage/control-robot.html)

任务配置还选择 embodiment、相机、深度/点云/分割等数据类型和域随机化。要显式开启并核验所需观测，不能因为环境支持这些数据就假定当前 obs 已提供。官方控制页目前还将策略评估指向 XPolicyLab，选择版本时应核对实际入口。[官方配置说明](https://robotwin-platform.github.io/doc/usage/configurations.html)、[官方控制说明](https://robotwin-platform.github.io/doc/usage/control-robot.html)

对本管线的接入工作：

- SAPIEN 不能直接调用我们基于 MuJoCo 的局部 Jacobian、接触查询与状态保存恢复；需要对应后端或隔离的等价规划/预检模型，并验证一致性。
- 按所选 embodiment 建模机器人与夹爪，确认单臂或双臂、基座、关节顺序、TCP 和实际动作规格。当前 Panda/cuRobo worker 的七关节模型不是双臂规划器。
- 若先验证一只手的抓放，明确选择、固定并建模另一只手；这只是受限实验，不能称为完整双臂任务支持。
- 交接、两手同时操作、双手持物需要额外调度、同步、附件和互碰处理，必要时扩展接口版本；当前可变长度 Schema 不代表这些能力已存在。
- 原生终止和任务成功应从所选任务/包装层读取，不能由 move_to 成功或规划器状态代替。

## RoboCasa

官方当前文档介绍 RoboCasa365，包含场景 split 与 seed 的 Gym 创建接口；仓库 README 说明 robosuite 依赖。不同历史版本和包装层需分别固定，不能把旧环境构造方式及动作布局直接当作当前版本保证。[官方概览](https://robocasa.ai/docs/build/html/introduction/overview.html)、[官方 Basic Usage](https://robocasa.ai/docs/build/html/introduction/basic_usage.html)、[官方 README](https://raw.githubusercontent.com/robocasa/robocasa/main/README.md)

对本管线的接入工作：

- 同属 robosuite/MuJoCo 生态时，部分几何和 IK 处理有复用机会，但需核对模型/索引、资产、相机、对象/fixture 注册和区域语义，不能直接替换 LIBERO 的 task_env。
- 从实际控制器读取完整动作布局。移动底盘、躯干或控制模式需要进入模型与执行逻辑；当前固定基座 Panda 模型不能只补动作尾部零就成为移动机器人模型。
- 若冻结底盘/躯干，只声明已验证的局部单臂任务范围；涉及导航或移动操作的任务不能据此计为支持。基座变化后更新场景和规划坐标，不能沿用旧 world_T_base。
- 给出完整 TCP 位置和朝向的执行方式；纯 XYZ 伺服不足以执行任意 GraspGen 六自由度抓法。
- 从原生任务取得最终判定，单独映射门/抽屉/旋钮、厨房 fixture、目标区域和复合顺序；未知技能明确不支持。

## 可选的本地 RPent 参考

本地 checkout 的 HEAD 为 `f11eee5910b74d132363a526ad385fa828ad4a26`，以下结论只描述这份源码，不是对所有官方版本的承诺。RPent 并非本管线必须依赖的中间件，也没有因此自动接上 GraspGen。

| 参考位置 | 本地实现 | 接入时的含义 |
| --- | --- | --- |
| [RoboTwin env_client](https://github.com/zhexuanxu/RPent/blob/f11eee5910b74d132363a526ad385fa828ad4a26/robots/robotwin/env_client.py) | 对 qpos/ee 分别校验 14/16 维动作，返回五元组并保存 episode_status | 这是该包装版本的规格；读取完整动作/双臂状态和真实 seed，不能当作任意 embodiment 的通用规格 |
| [RoboTwin primitives](https://github.com/zhexuanxu/RPent/blob/f11eee5910b74d132363a526ad385fa828ad4a26/robots/robotwin/primitives.py) | move_to 调原生 planner 得 qpos 路点；finish 以新查询的 eval_success 为准 | 不要把原生重新规划的路径归为我们 cuRobo 的输出，也不要用工具请求的 success 覆盖任务结果 |
| [RoboCasa env_server](https://github.com/zhexuanxu/RPent/blob/f11eee5910b74d132363a526ad385fa828ad4a26/robots/robocasa/env_server.py) | 使用 PandaOmron，get_action_spec 描述机械臂、夹爪、躯干、底盘和控制模式；check_success 调原生 _check_success | 动作维数和语义需实际读取；这不是原 LIBERO 七维 OSC 的同一个机器人配置 |
| [RoboCasa env_client](https://github.com/zhexuanxu/RPent/blob/f11eee5910b74d132363a526ad385fa828ad4a26/robots/robocasa/env_client.py) | 提供像素对齐的深度反投影 world_map | 可参考相机转换，但仍需目标定位/掩码、时间一致性和来源标记 |
| [RoboCasa primitives](https://github.com/zhexuanxu/RPent/blob/f11eee5910b74d132363a526ad385fa828ad4a26/robots/robocasa/primitives.py) | move_to 是 XYZ 伺服并保持原有朝向；current_state_dict 明确不提供物体坐标 | 不能丢弃 GraspGen 朝向；teacher 模式若需要物体真值，应另行显式提供，不能读取不存在字段 |

这些包装中可能还有 VLA 或其他工具。若接入试验使用它们代替本管线的一部分，应在配置和结果里单列实际后端，不把其成功直接算作 GraspGen＋cuRobo 的效果。

## 两者共同的最低交付

接入方交付所选版本与机器人说明、新提供方和真实启动入口、坐标/模型/动作验收、原生成功映射、单任务连续视频与失败记录，再按 [交付标准](RESULTS_HANDOFF.md) 扩大范围。本轮只提供这些接入说明，不安装或运行两个 benchmark。
