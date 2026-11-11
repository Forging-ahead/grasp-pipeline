# 基线与实验产物

## 权威版本

基线为最终 v10 四卡服务器完整 130 任务测评所用源码。来源是 [research-source.zip](../../outputs/libero-final130-four4090-20261111/delivery/research-source.zip) 中的 `versions/v10/`，本次原样复制到 [baseline/v10](../baseline/v10/)。50 个文件逐字节相同，其中 39 个 Python 文件的哈希与归档内 `FROZEN_SOURCE.json` 一致。

| 项目 | 位置或指纹 |
| --- | --- |
| 本次冻结参考 | `pipeline-organization/baseline/v10/` |
| 原开发目录 | `work/libero-policy-broad-20261111/` |
| 原测评工具目录 | `work/libero-final130-four4090-20261111/` |
| 最终交付结果 | `outputs/libero-final130-four4090-20261111/delivery/` |
| 服务器实测快照 | `/mnt/public/changwei/Graspgen/libero-final130-four4090-20261111/versions/v10/`，服务器 SSH 端口 40596 |
| `run_task.py` SHA256 | `04c2db93611eefa47e4cbf48b80c996242b7fad86e291d069da90c097eb29e22` |
| `research-source.zip` SHA256 | `17567c081f0cf62f8536f2d5231d489a9e8aa03e1a28226cb0c2d192607a5977` |
| 任务清单 SHA256 | `c313169626a5b681cffde7e5e011454ba46ea46c7aa5fdd47a3431e45fcf39cd` |

开发目录中的 39 个 Python 文件与该基线一致，但开发目录整体不等于冻结快照：`ALL_TASKS.json` 不同，且根目录没有最终测评的 `FROZEN_SOURCE.json`。后续整理以本次冻结源码和配置为起点，不能直接用开发目录任务清单替换。

冻结副本中的桥接 `README.md` 仍保留早期单任务说明，部分内容已过时；`FIX_RELEASE.json`、`SOURCE_SHA256.json` 等也保留其当时的含义。它们作为历史文件原样保存，当前范围以本组说明、最终驱动和最终测评记录为准。

## 实验结果

最终测评每个任务只跑一个固定初态，`init_state_id=0`、`seed=0`，共 130 次。运行期间未更改策略源码和配置逻辑，没有合并不同版本的历史成功记录。

| 口径 | 数量 | 含义 |
| --- | --- | --- |
| 官方最终条件满足 | 88/130 | 最后状态满足原始任务目标 |
| 完整管线成功 | 86/130 | 管线完成、最终条件满足且执行动作数大于 0 |
| 排除已知顺序错误的候选成功轨迹 | 85/130 | 进一步排除一条“先堆叠再搬运”顺序不符合指令的记录 |
| 管线未完整成功 | 44/130 | 保留失败视频和诊断；其中 17 个只进行了规划或预检，27 个已有物理动作 |

这组数字描述当前固定初态的工程测评，不是多初态 benchmark 成功率。85 条也还需要训练数据质量检查，不能直接称为全部合格的 SFT 数据。

已知顺序问题任务为 `libero_90_LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray`：策略先把底碗放到托盘，再叠上另一个碗；最后状态满足目标，但不满足“先叠再整体搬运”的顺序要求。第一阶段保留原记录，不重新分类为新结果。

## 结果入口

- [含 130 个视频的独立 HTML](../../outputs/libero-final130-four4090-20261111/delivery/LIBERO130%20v10复测报告_含全部视频.html)
- [原始结果 ZIP](../../outputs/libero-final130-four4090-20261111/libero-final130-v10-results.zip)
- [全部任务结果](../../outputs/libero-final130-four4090-20261111/delivery/RESULTS.json)
- [16 类统计](../../outputs/libero-final130-four4090-20261111/delivery/CATEGORY_SUMMARY.json)
- [失败原因统计](../../outputs/libero-final130-four4090-20261111/delivery/FAILURE_SUMMARY.json)
- [测评协议](../../outputs/libero-final130-four4090-20261111/delivery/PROTOCOL.json)
- [产物路径与指纹索引](../audit/ARTIFACT_INDEX.json)

本地 HTML 名称为 `LIBERO130 v10复测报告_含全部视频.html`，归档内名称为 `LIBERO130任务最终v10复测报告_含全部视频.html`。二者内容 SHA256 均为 `a487ff8e3ce0b8587d7296f63a579cffebf8060b88899797e8956d56e32f60ff`；未更改或重生成报告。

## 历史目录边界

`libero-30`、`libero26-*`、`libero-fix-*`、`libero-placement-fix-*`、`libero-counterfactual-*`、`libero-sixfix-*`、`libero-policy-generalize-*` 和旧 `libero-final130-20261111` 等目录保存早期开发、诊断、分类或部分测评。它们继续保留，不能作为本次 v10 成绩的补集。

`RPent/` 是另一仓库的 `rpent-ppo` checkout。现有多环境包装可作为以后文档中的参考，但不意味着本 GraspGen 管线已经在这些环境上接通或取得成绩。本阶段没有修改 RPent。

完整路径盘点见 [WORKSPACE_INDEX.json](../audit/WORKSPACE_INDEX.json)。
