# GraspGen 管线整理入口

五个整理阶段均已完成：基线盘点、代码整理、接入接口定义、外部实施指南和验证交付。这里保存最终 v10 的独立副本及可核验清单；整理后的独立代码入口为 [grasp-pipeline/README.md](../grasp-pipeline/README.md)。原开发目录、冻结副本、测评归档和视频均保留。

## 从这里开始

| 想了解什么 | 入口 |
| --- | --- |
| 全部整理阶段和当前状态 | [PLAN.md](PLAN.md) · [在线进度页](https://chatgpt.com/space/page_4469bb3185288191a60707ef665c2f7a) |
| 整理后的代码、入口与实际改动 | [代码入口](../grasp-pipeline/README.md) · [第二阶段改动说明](../grasp-pipeline/docs/CHANGES_PHASE2.md) |
| 接入字段、方法、能力要求与参考 | [第一版接口](../grasp-pipeline/docs/interfaces/README.md) · [第三阶段交付](../grasp-pipeline/docs/CHANGES_PHASE3.md) |
| 接入方需要提供、修改和验证什么 | [外部实施指南](../grasp-pipeline/docs/integration/README.md) · [第四阶段交付](../grasp-pipeline/docs/CHANGES_PHASE4.md) |
| 如何部署、检查与转交代码 | [部署说明](../grasp-pipeline/deploy/README.md) · [第五阶段交付](../grasp-pipeline/docs/CHANGES_PHASE5.md) · [源码 ZIP](../deliverables/grasp-pipeline-v10-handoff-20261112.zip) |
| 当前哪份代码有效、实验结果对应哪一版 | [基线与实验产物](docs/BASELINE.md) |
| 各组件做什么、调用顺序是什么 | [代码和数据流程](docs/CODE_MAP.md) |
| 配置在哪里、哪些值实际生效、需要哪些环境 | [配置与依赖](docs/CONFIG_AND_RUNTIME.md) |
| 外部 benchmark 接入需要提供什么、当前耦合在哪 | [接入边界盘点](docs/INTEGRATION_BOUNDARIES.md) |
| 如何核验和复现基线 | [基线核验与运行说明](docs/REPRODUCING_V10.md) |

## 本目录

```text
pipeline-organization/
  PLAN.md                  整理计划和完成标准
  PROGRESS.json            机器可读进度
  baseline/v10/            从最终测评源码包逐字节复制的 50 个文件
  docs/                    第一阶段说明
  audit/                   源码、配置、依赖、产物和验证清单
  tools/inventory_baseline.py  本地盘点与只读核验工具
```

50 个基线文件包含 39 个 Python 文件，其中 25 个为运行代码、14 个为测试文件；其余为配置、包元数据和原有说明。`baseline/v10/` 是冻结参考，后续重构应在另一个工作副本进行。

本阶段验证通过：39 个 Python 文件和开发目录一致，50 个基线副本与归档一致，3270 个交付文件内容哈希匹配，130 个任务身份与 16 类统计一致。报告在本地被重命名，核验按完全相同的内容哈希识别，映射记录在 [ARTIFACT_INDEX.json](audit/ARTIFACT_INDEX.json)。

第二阶段完成了独立目录、模块分组、配置和入口整理。原版 86 项 CPU 测试与整理版 90 项测试全部通过；1040 组有效配置无差异。记录见 [CPU 验证](../grasp-pipeline/migration/CPU_VALIDATION.json)与[源码及参数核验](../grasp-pipeline/migration/PRESERVATION_CHECK.json)。

第三阶段定义九类方法接口和 14 类数据包，提供真实 LIBERO 记录样例、源码映射与离线校验；该阶段 113 项 CPU 测试全部通过，记录见 [第三阶段验收](../grasp-pipeline/migration/phase3/VALIDATION.json)。

第四阶段已给出输入与责任清单、21 个现有源码定位点、四个配置/接口样例、G0–G8 验收和结果交付说明，核对了 RoboTwin/RoboCasa 官方资料与固定 RPent 参考。文档、示例、轻量 CLI 和独立代码目录检查通过，原 85 个已验收文件未变。

第五阶段补齐三环境部署与只读启动前检查，独立解压候选包的 121 项 CPU 测试与三个组件帮助入口通过。最终 ZIP 的 224 个内容文件、50 个冻结参考、文档链接和实测源码一致性核验通过，见[归档验收](../deliverables/grasp-pipeline-v10-handoff-20261112.validation.json)。

目前尚未新增通用环境适配器，旧驱动没有切换到新接口，也未在新环境上运行轨迹。CUDA/EGL、模型推理和完整任务仍需在实际部署机器另行验收；文档完成不表示换一个环境名就能运行。
