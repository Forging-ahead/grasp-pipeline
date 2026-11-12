# 从 libero 分支继承实现，再适配新环境

**接入其他 benchmark 必须以本仓库的 `libero` 分支为开发基线。已有的 code policy、GraspGen 调用、cuRobo 规划、接触预检和控制修复应优先直接复用；不要只参考文档流程，再从零写一套简化管线。**

迁移的主要工作是把新环境的任务、观测、机器人、碰撞场景和动作接口接到已有实现上。新环境确实存在不同的机器人、物理引擎或技能时，再新增对应后端，并记录与基线的差异。

## 建立开发分支

新 checkout 示例（将 `bench/my-benchmark` 换成实际分支名）：

```bash
git clone --branch libero https://github.com/Forging-ahead/grasp-pipeline.git grasp-pipeline-workspace
cd grasp-pipeline-workspace
git switch -c bench/my-benchmark
cd grasp-pipeline
python3 run.py --help
python3 tools/verify_preservation.py
```

已有 checkout 则先 `git fetch origin`，再从 `origin/libero` 创建自己的开发分支。记录实际起点 commit；不要使用空项目作为起点，也不要覆盖 `libero` 的任务配置或冻结参考。已有 benchmark 分支可以作为额外参考，其代码和结果不自动成为新环境的验收证据。

## 哪些实现应当继承

下列路径相对运行目录 `grasp-pipeline/`。优先采用直接调用、继承或薄适配器，让成熟规则继续由原函数执行；确实必须移植时，保留来源、参数和行为对照。

| 已有实现 | 迁移要求 |
| --- | --- |
| `models/graspgen_worker.py`、`common/rpc.py` | 复用模型调用、点云输入和进程隔离；更换夹爪时单独核对权重与参考点 |
| `policy/evaluate.py` 的 `PickPlace.select_candidate`、`recover_native_descent` | 继承候选排序、多轮搜索、接近/下探与恢复逻辑，不缩成“最高分抓法只试一次” |
| `planning/grasp_contact.py`、`planning/cartesian.py` | 保留抓取接触预检、局部接触路径与无副作用预演；同引擎先做模型/索引映射 |
| `PickPlace.transfer_with_alternatives`、`policy/transfer.py` | 复用搬运的朝向、高度和备选搜索，接入新目的地几何，不能只规划一个固定终点 |
| `policy/placement.py`、`PickPlace.place_in_container_static` | 复用开口、物体脚印、释放高度、两段预检和实测状态重算，不退化成“容器中心加固定高度” |
| `planning/planner_worker.py`、`planning/planning_geometry.py` | 复用 cuRobo 的请求、场景/持物和失败处理；按新机器人的真实模型验证 FK、限位与碰撞 |
| `control/controller.py` 的 `Controller.move_to` | 保留六自由度跟踪、限位恢复、反馈、预算和分支保护；将输出映射为新环境原生动作 |
| `policy/task_policy.py`、`policy/skills.py` | 保留适用的任务调度与技能规则；仅替换任务定义、实体/区域与机构语义的读取，不凭技能同名宣称已迁移 |
| 原生成功、失败分类与记录逻辑 | 保留连续回合、真实动作、阶段决策、版本和失败视频；使用新环境的原始成功判定 |

这里的“复用”不等于原函数无需适配即可运行。旧实现有直接访问 `sim`、关节、控制器和任务对象的地方；应按[文件修改清单](FILE_CHANGES.md)逐项提供兼容输入或后端，验证实际调用链，不能只实现一个空接口。

## 哪些内容需要新环境适配

- 生命周期、官方任务/初态、实体与目标区域、成功/终止规则。
- 同步 RGB-D、目标掩码、米制点云、相机标定与观测来源。
- 真实机器人关节、基座、TCP、夹爪、运动学与碰撞模型。
- 场景、支撑面、持物附件和阶段接触许可。
- 原生控制动作、控制周期，以及需要时的局部 IK/预检后端。

同一物理引擎和相似机械臂不代表这些输入相同；也不能因此直接重写整套策略。先定位差异，补映射，再检查原规则能否直接运行。新环境缺少的技能需要另行实现和验证，不能用抓放代替门、旋钮或其他受约束操作。

## 参数和差异如何记录

首个迁移对照应继承 LIBERO 的有效共享参数与搜索规则。TCP 标定、动作尺度、机器人限位等必须由新环境实测；不要把 LIBERO 的机器人常量照抄成通用值，也不要复制某个任务的特殊覆盖作为全部任务默认值。

每项变更记录：原文件/函数、原参数来源、新环境差异、处理方式（直接复用/适配/新增）、新调用位置和验证证据。建议用下表随代码提交：

| 原实现及指纹 | 新调用位置 | 处理方式 | 必须改变的原因 | 验证证据 |
| --- | --- | --- | --- | --- |
| 例如 `PickPlace.select_candidate` | 新环境 selector | 继承原函数 | 仅适配环境查询 | 候选/接触检查与连续回合记录 |

未记录的简化实现不能称为“沿用成熟 LIBERO 规则”。如果确需改共享算法或预算，建立新版本，保留对照并分开统计。

## 验证与交付

1. 保存基线 commit、复用关系和有效配置，先运行原有静态保真与回归检查。
2. 检查坐标、FK、夹爪符号、动作尺度、场景几何和预演是否改变真实状态。
3. 用同一连续回合验证接近、闭合、抬升、搬运、放置和撤离，保留真实阶段证据。
4. 抓放之外的技能分别验证；明确区分“未实现/未启用”“已尝试失败”和“完整成功”。只初始化场景的任务不算操作测试。
5. 冻结一版代码后统一运行任务矩阵，保留成功和失败视频，不混合不同版本或不同预算的最优结果。

原有检查入口是 `python3 tools/verify_preservation.py`、`python3 tools/check_integration_docs.py`，以及在准备好开发依赖后执行的 `python3 -m pytest -q tests`。这些检查不替代新 benchmark 的 GPU 物理实验。具体输入、接线和验收见[接入指南](README.md)。
