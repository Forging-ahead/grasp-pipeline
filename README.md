# Grasp Pipeline

接入新的 benchmark，请从本 `libero` 分支创建开发分支，继承已有的策略、规划、接触预检和控制实现，再适配新环境。不要另起空项目重写简化管线；具体要求见[基线继承说明](grasp-pipeline/docs/integration/BASELINE_FIRST.md)。

基于 GraspGen、cuRobo 和 LIBERO / MuJoCo 的机器人操作管线：code policy 计算各阶段目标位姿，GraspGen 生成并评分抓取位姿，规划与控制模块负责路径、执行和结果记录。目前运行链支持 LIBERO、单 Panda；其他 benchmark 的适配接口和实施指南已经整理，运行适配器尚未实现。

当前进度：五个整理阶段已完成。2026-11-13 用本 `libero` 分支的冻结 v10 策略完整重跑 130 项：86 项流程与官方终态通过（66.2%），剔除 1 项已知指令顺序问题后为 85 条候选轨迹。130 段成功与失败视频均已保存并通过完整性校验，策略与参数保持不变。详见 [全量复跑记录](grasp-pipeline/docs/LIBERO130_20261113.md) 与 [机读结果](grasp-pipeline/migration/LIBERO130_20261113.json)。

- [整理入口与第一阶段交付](pipeline-organization/README.md)
- [整理后的自有代码与运行入口](grasp-pipeline/README.md)
- [接入接口、字段与 LIBERO 参考](grasp-pipeline/docs/interfaces/README.md)
- [外部 benchmark 接入实施指南](grasp-pipeline/docs/integration/README.md)
- [从 libero 分支继承实现，再适配新环境](grasp-pipeline/docs/integration/BASELINE_FIRST.md)
- [部署与启动检查](grasp-pipeline/deploy/README.md) · [第五阶段交付](grasp-pipeline/docs/CHANGES_PHASE5.md)
- [整理计划与进度](pipeline-organization/PLAN.md)
- [最终 v10 源码基线](pipeline-organization/baseline/v10/)

## 获取代码后

保留仓库中的两个代码目录及其相对位置；冻结基线供版本核对，实际运行入口在 `grasp-pipeline/`。

```bash
cd grasp-pipeline
python3 run.py --help
python3 run.py list
python3 tools/verify_preservation.py
python3 tools/check_integration_docs.py
```

上述命令用于配置预览和静态核对，不启动任务。实际仿真与模型运行需要 Linux NVIDIA GPU、三个外部 Python 环境、固定上游源码、模型权重和仿真资产，详见[部署说明](grasp-pipeline/deploy/README.md)。机器路径填写在被 Git 忽略的 `grasp-pipeline/configs/runtime/paths.local.json` 中。

## 版本管理范围

本仓库跟踪整理后的源码、任务配置、测试、部署与接入文档，以及冻结 v10 基线和整理记录。模型权重、视频、实验输出、完整环境、交付归档和独立上游 checkout 不纳入 Git；原 130 任务的视频报告与源码交付 ZIP 需要单独提供。历史结果与本轮整理检查的区别见[版本与验证范围](grasp-pipeline/docs/RELEASE_VALIDATION.md)。

原工作区中的 `work/` 保存历史开发脚本，`outputs/` 保存实验产物，`deliverables/` 保存交付归档；这些目录保留在本地并由 `.gitignore` 排除。`RPent/` 是独立的上游项目 checkout，不是本仓库的运行依赖。上游资源遵循各自许可，本仓库整理不改变其授权。
