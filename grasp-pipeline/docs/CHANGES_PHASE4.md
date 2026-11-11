# 第四阶段交付

外部接入指南位于 [docs/integration/README.md](integration/README.md)，包含输入与能力、现有文件改动、配置样例、调试验证、结果交付以及 RoboTwin/RoboCasa 差异，共七篇正文文档。

## 本次补齐的内容

- 明确我们现有组件和接入方责任，指出九类 Protocol 还没有与当前运行链接通。
- 列明 21 个现有源码定位点及五类拟新增路径，解释当前驱动、配置、机器人模型、场景、局部 IK、接触、控制和原生成功如何接线。
- 提供四个 JSON 样例，分别标明当前 LIBERO 路径格式、设计单、TaskSpec 与未实现 Capabilities，避免把模板当作运行注册。
- 给出 G0–G8 分层调试、常见失败排查、任务覆盖/成功统计与最终交接标准。
- 核对官方资料和本地 RPent 固定 commit，分别说明双臂/SAPIEN、移动底盘/动作布局、目标位姿和实际规划后端的适配工作；没有实现两个 benchmark。
- 补充训练数据需要的动作前观测、教师请求参数和动作后实测结果，区分 code policy 标签与 VLM 决策。
- 新增只读文档检查工具，保存文档链接、源码定位、配置格式、轻量 CLI 与两个公开示例的 Schema 验证。

## 保持不变

第三阶段验收的 85 个源码、配置和参考文件保持原样，本次没有改策略、参数、接口 Schema 或运行驱动；只新增指南、样例、来源记录和文档检查工具。原先 113 项 CPU 测试仍是第三阶段验证记录，本轮没有把它们冒称为新环境验证，也未启动模型/GPU/任务轨迹。

本次验收见 [DOCUMENTATION_QA.json](../migration/phase4/DOCUMENTATION_QA.json)与[SCHEMA_QA.json](../migration/phase4/SCHEMA_QA.json)。后续部署与交付见[第五阶段说明](CHANGES_PHASE5.md)，当前仍没有其他 benchmark 的运行适配器。
