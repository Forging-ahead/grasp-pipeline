# 第五阶段交付

第五阶段完成部署说明、只读启动前检查、独立交付包和必要验收。运行策略、参数、驱动和第一版接口 Schema 保持原样；第三阶段的 85 个受检文件继续按原指纹核对。

## 新增内容

- [部署与启动检查](../deploy/README.md)：三个环境、外部仓库/资产/权重、六路径、wrapper、诊断和完整任务命令，以及从零安装与 GPU 验收的边界。
- [原环境依赖](../deploy/reference/dependencies.v10.json)、[本轮环境观察](../deploy/reference/RUNTIME_INVENTORY.json)和[固定源码记录](../deploy/reference/SOURCES.json)：不把 observed distribution 列表当成安装锁文件，嵌入的 GraspGen 源码树与 fork commit 分开记录。
- [check_deployment.py](../tools/check_deployment.py)：路径检查，可选三个环境 CPU 导入、核心版本及实际 YAML 引用的两个权重指纹检查；错误明确返回失败。
- [build_handoff.py](../tools/build_handoff.py)：创建新目录和 ZIP，附 50 个冻结参考文件、逐文件大小/SHA256 和归档指纹；拒绝覆盖旧交付，支持解压后只读核对。
- [八项部署与交付工具测试](../tests/test_deployment_tools.py)：覆盖失败分类、禁止路径夹带策略参数、禁止把路径通过当作 GPU 验收、包内文件变动及覆盖保护。
- [版本与验证范围](RELEASE_VALIDATION.md)：分开说明历史 130 任务成绩与本轮源码/CPU/部署/交付检查。

## 本轮实际验证

在原授权服务器的新解压目录中，使用现有依赖运行全部 **121 项 CPU 测试**，无失败、错误或跳过。模型和规划 worker 的 `--help`、仿真 CLI、接口 Schema、130 条历史结果样例、机械迁移及 1040 组参数核对均通过。测试前后的包内容清单未变。

本地交付源码、配置与参考的 92 个文件和通过测试的副本一致，原 85 个第三阶段文件未变。另在三个实际角色环境中，核心导入与版本、Panda YAML 和两个实际权重指纹通过；检查隐藏 GPU，未初始化 CUDA。

原有 3270 个交付文件、50 个冻结副本与 39 个原 Python 文件重新核对通过。详细记录与原日志见 [migration/phase5](../migration/phase5/)，总览见 [VALIDATION.json](../migration/phase5/VALIDATION.json)。

最终 ZIP 的精确内容与指纹验收放在归档旁的 `.validation.json` 和 `.sha256`，避免把包含自身哈希的文件再打入自身。包内 `MANIFEST.json` 列出其余每个文件的内容指纹。

## 仍需另行实施和验证

没有从零安装完整环境，没有运行模型推理、EGL 场景或新的完整任务；本轮不产生新成功率。RoboTwin、RoboCasa 适配仍由接入方按第四阶段文档实现；现有运行链尚未切换到九类 Protocol。权重、上游源码、资产、三个环境和原任务视频不在源码交付包中。
