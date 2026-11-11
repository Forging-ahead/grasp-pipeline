# 组件结构与文件映射

## 源码现在在哪里

以下文件原来都在一个 `graspgen_libero_bridge/` 平级目录中。现在按职责分组，文件内算法和数值常量保持原样，跨组件导入已改到新位置。

| 新目录 | 文件 | 职责 |
| --- | --- | --- |
| `policy/` | `task_policy.py`、`evaluate.py`、`skills.py`、`placement.py`、`transfer.py`、`surface_packing.py` | 任务目标与阶段顺序、抓放流程、接触技能调度、搬运和放置候选 |
| `models/` | `graspgen_worker.py` | 加载 Panda 预训练模型，生成和评分抓取位姿 |
| `planning/` | `planner_worker.py`、`planning_geometry.py`、`cartesian.py`、`grasp_candidate.py`、`grasp_contact.py` | cuRobo、机器人碰撞表示、MuJoCo 局部 IK、候选几何和接触预检 |
| `control/` | `controller.py` | OSC 闭环跟踪、限位和接触监测、状态动作视频记录 |
| `perception/` | `pointcloud.py` | 深度及掩码反投影、点云采样和几何输入 |
| `adapters/libero/` | `env_adapter.py`、`scene.py`、`regions.py`、`placement_geometry.py`、`articulation.py` | LIBERO 环境与状态、仿真场景、目标区域、关节轴和把手几何 |
| `common/` | `frames.py`、`rpc.py`、`errors.py` | 原坐标转换、工作进程通信与失败类型 |
| `common/` | `layout.py` | 新增的 checkout 路径与递归源码指纹；不参与目标位姿计算 |
| `runtime/` | `config.py`、`cli.py`、`legacy_cli.py` | 新配置加载、统一入口以及迁移后的旧单任务命令行 |
| `contracts/` | `ports.py`、`validation.py` | 第三阶段接入约定及离线校验；不由现有 actor 导入 |

完整任务驱动从原根目录 `run_task.py` 移到 `apps/libero/run_task.py`。原 14 个测试文件移到 `tests/v10/`，只调整导入和 RPC 子进程搜索路径；原测试断言保留。新增配置入口测试位于 `tests/test_runtime_config.py`。

## 运行顺序

`run.py` 解析命令，读取部署解释器路径，再启动 `apps/libero/run_task.py`。驱动加载指定任务的原配置与原覆盖参数，创建 LIBERO 环境、编排目标阶段，依次调用抓放或接触技能。

GraspGen 子进程现在由 `graspgen_libero_bridge.models.graspgen_worker` 启动，cuRobo 子进程由 `graspgen_libero_bridge.planning.planner_worker` 启动。两个进程仍使用原解释器、原请求协议和原模型。启动时传入的源码搜索路径统一指向 `src/`。

控制路径保持原分工：全局接近和搬运主要使用 cuRobo，接触阶段和部分备用路径使用 MuJoCo 局部 IK，最后由 OSC 执行。新的目录分组没有把算法替换为另一规划器。

## 使用方需要调整的导入

原来的平级模块导入改为分组后的导入，例如：

```python
from graspgen_libero_bridge.policy.task_policy import schedule
from graspgen_libero_bridge.policy.evaluate import PickPlace
from graspgen_libero_bridge.control.controller import Controller
from graspgen_libero_bridge.adapters.libero.env_adapter import EnvAdapter
```

本目录内所有调用方、测试和工作进程模块名都已同步修改。原开发目录没有改动，继续使用它时仍按其旧结构导入。所有旧位置到新位置的对应关系及逐项替换见 `migration/V10_RELOCATION.json`。

## 当前仍存在的环境依赖

本阶段完成目录和启动边界整理，没有把全部环境访问抽象为通用接口。`planning/cartesian.py`、`planning/grasp_contact.py`、`control/controller.py` 和部分策略仍直接使用 MuJoCo、LIBERO 数据或 Panda 假设。

因此 `adapters/libero/` 是已明确归类的环境相关代码，不代表其他目录已完全独立于仿真器。第三阶段已定义 [统一数据和方法约定](interfaces/README.md)，但尚未据此重接旧驱动，也未实现其他 benchmark。
