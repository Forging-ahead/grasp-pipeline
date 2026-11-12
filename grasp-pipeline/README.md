# GraspGen 管线

这是从最终 v10 整理出的独立工作目录。它保留原有 code policy、GraspGen、cuRobo、MuJoCo 局部 IK 和 OSC 执行逻辑，将源码、配置、入口、测试与历史测评工具分开。

接入新 benchmark 时，从 `libero` 分支创建自己的分支，优先直接调用或继承这些成熟实现，只适配环境差异；不要从零重写简化策略。开发起点、复用要求和差异记录见[基线继承说明](docs/integration/BASELINE_FIRST.md)。

目前实际环境仍是 LIBERO、单 Panda 和原有夹爪。接入约定见 [第一版接口](docs/interfaces/README.md)；现有驱动尚未改为通过这些接口调用，本轮没有实现 RoboTwin 或 RoboCasa 适配。

## 目录

```text
grasp-pipeline/
  run.py                         统一启动入口
  apps/libero/run_task.py         连续任务驱动
  src/graspgen_libero_bridge/
    policy/                      技能编排、抓放、搬运和放置策略
    models/                      GraspGen 工作进程
    planning/                    cuRobo、局部 IK 与候选预检
    control/                     OSC 跟踪、监测和记录
    perception/                  点云处理
    adapters/libero/              LIBERO 状态、场景、区域和关节几何
    common/                      坐标、RPC、错误和路径工具
    runtime/                     配置加载和命令行
  configs/                       任务、原策略覆盖、部署路径与旧单任务示例
  tests/                         原版测试及新的配置入口测试
  tools/                         核验工具与独立保存的旧测评脚本
  migration/                     文件迁移、参数保持和 CPU 验证记录
  docs/                          结构和改动说明
```

## 常用入口

在本目录运行，以下命令不加载模型或创建环境：

```bash
python3 run.py --help
python3 run.py list
python3 run.py config libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket --gpu 0 --init 0
python3 run.py run libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket --gpu 0 --init 0 --tag example --dry-run
```

`--dry-run` 只显示实际命令、配置和输出路径。去掉它才会使用 `configs/runtime/paths.server.json` 中的仿真解释器运行任务；需要已准备好的 Linux、CUDA、LIBERO、模型权重和三个依赖环境。开发主机的系统 Python 可以查看配置，不能据此认为已具备运行仿真的依赖。

其他部署位置：复制 `configs/runtime/paths.server.json` 为 `configs/runtime/paths.local.json`，只改六个绝对路径，然后给 `config` 或 `run` 加 `--paths configs/runtime/paths.local.json`。该文件不允许写入策略参数。

真实运行输出默认保存到 `runs/<tag>/<task_key>/`，已有目录会拒绝复用；可用 `--output-root /absolute/output/path` 改变输出根目录。每阶段和原始结果记录结构保留。

原单任务诊断入口单独保留：

```bash
python3 run.py legacy doctor --config configs/examples/single_task.v10.json
```

该示例是旧单任务配置，不能作为完整 130 任务的同配置运行。`legacy run` 同样保留原显式抓放接口；完整任务使用上面的 `run <task_key>`。

推荐从 checkout 的 `run.py` 启动，也支持在配置了 `src` 的环境里使用 `python -m graspgen_libero_bridge`。包元数据用于源码 checkout 的 editable 安装；配置仍保存在 checkout 中，当前不提供脱离 checkout 的完整 wheel 部署包。

## 阅读与验证

- [部署依赖、路径与启动检查](deploy/README.md)
- [交付版本与验证范围](docs/RELEASE_VALIDATION.md)
- [组件结构与原文件映射](docs/LAYOUT.md)
- [配置来源与优先级](configs/README.md)
- [第二阶段具体改动](docs/CHANGES_PHASE2.md)
- [测试与验证范围](docs/TESTING.md)
- [接入接口、字段和 LIBERO 参考](docs/interfaces/README.md)
- [第三阶段具体交付](docs/CHANGES_PHASE3.md)
- [外部 benchmark 接入实施指南](docs/integration/README.md)
- [第四阶段具体交付](docs/CHANGES_PHASE4.md)
- [第五阶段部署、打包与验收](docs/CHANGES_PHASE5.md)
- [迁移逐项记录](migration/V10_RELOCATION.json)
- [源码与参数核验结果](migration/PRESERVATION_CHECK.json)

```bash
python3 tools/verify_preservation.py
```

该只读核验需要工作区中的冻结基线，或通过 `--baseline /path/to/v10` 显式指定。它检查 39 个原文件的迁移改动、内部导入、原任务清单及 1040 组有效配置。完整依赖和权重未包含在本目录中。
