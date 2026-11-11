# GraspGen 管线交付包

本包整理最终 v10 的 code policy、GraspGen/cuRobo 调用、MuJoCo 局部 IK 与 OSC 执行代码，并附冻结基线、配置、接口、接入文档和验证记录。策略算法与参数保留。当前运行环境仍是 LIBERO、单 Panda，其他 benchmark 的运行适配器尚待接入方实现。

## 从这里开始

1. 阅读[部署与启动检查](grasp-pipeline/deploy/README.md)，准备外部依赖和六个部署路径。
2. 阅读[外部 benchmark 接入指南](grasp-pipeline/docs/integration/README.md)，按输入清单、文件修改和分层验收实施。
3. 查看[版本与验证范围](grasp-pipeline/docs/RELEASE_VALIDATION.md)，区分历史任务成绩、本轮 CPU/交付检查和待做的 GPU 验收。

在本目录只读核对包内容：

```bash
python3 grasp-pipeline/tools/build_handoff.py --verify .
cd grasp-pipeline
python3 tools/verify_preservation.py
python3 tools/check_integration_docs.py
python3 run.py --help
```

`MANIFEST.json` 列出文件大小和 SHA256。下载时同时取得 ZIP 旁的 `.sha256` 文件核对归档。冻结参考位于 `pipeline-organization/baseline/v10/`，请保留它以运行版本保持检查。

## 包内和包外

包内包含自有源码、测试、原任务参数、接口与样例、部署说明、已记录的验收资料和 50 个冻结参考文件。`grasp-pipeline/` 是代码根目录；`pipeline-organization/baseline/v10/` 供核对，不是新的执行入口。

权重、三个完整 Python 环境、上游仓库、仿真资产和原 130 个视频未打入本源码包。它不包含新的 benchmark 成绩。原 130 任务视频报告保留在原实验交付目录，需要查看视频时单独提供该报告。

上游资源由部署方从对应项目获取并遵守其适用许可；本次整理不改变这些资源的授权。RPent 是可选参考，不是本包运行的必需依赖。

## 交接时需要提供的证据

新环境至少提供固定版本与机器人配置、实际观测来源、完整控制链、原生成功判定，以及一条连续执行的视频和记录。所有失败一并保留。接口字段通过、规划器返回路径和真实完整任务成功分别记录，不能互相替代。
