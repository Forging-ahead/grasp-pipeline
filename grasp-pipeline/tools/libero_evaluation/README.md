# 整理版 LIBERO 130 项复跑

本目录为 2026-11-13 的 `libero` 分支完整复跑增加独立调度、审核与报告工具。实际运行的 actor 固定为提交 `6d8cd7d7e6a34878bb71a0f5e61007e3f81ac405` 的源码快照；没有修改抓取、规划、控制或任务配置。原 `legacy_evaluation/` 保持原样。

先复制该提交到独立源码目录，准备已有三个运行环境和六项部署路径。`reference/taxonomy.v10.json` 只保存同一 130 项的任务定义与 16 类分类，不含历史结果。预算仍为初态 0、种子 0、最多 5000 步；96 项墙钟 1200 秒、34 项 1800 秒，超时预留退出清理。

```bash
python tools/libero_evaluation/run_all.py \
  --source /data/frozen-source/grasp-pipeline --paths /data/paths.local.json \
  --taxonomy tools/libero_evaluation/reference/taxonomy.v10.json \
  --output /data/libero130/evaluation --commit 6d8cd7d7e6a34878bb71a0f5e61007e3f81ac405 \
  --gpus 0 1 2 3 --sessions-per-gpu 2 --validate-only
# 去掉 --validate-only 才执行；输出目录必须尚不存在。
python tools/libero_evaluation/watch.py /data/libero130/evaluation --watch
```

`GPU_SLOTS.json` 允许仅调整空闲卡的并发，每卡最多 4 项、总计最多 16 项；变动写入 `SCHEDULING_HISTORY.json`。每项启动和结束均核对 actor 指纹；结果必须完整匹配运行模块、驱动、任务和初态。错版本、缺结果、超时、零动作以及部分子目标满足不会被计为完整成功。没有外层策略结果重试，失败保留分母。

模型权重和上游源文件的 `EXTERNAL_SOURCE_BEFORE.json`（路径到 SHA256 的映射）、部署检查及回归记录放在实验根目录（evaluation 的上一层）。同处还需保留原始 Git 归档 `libero-6d8cd7d-20261113.tar.gz`，供交付时恢复准确源码树；本轮服务器已准备这些前置记录。只在全部 130 项结束后执行：

```bash
python tools/libero_evaluation/package_results.py /data/libero130/evaluation
python tools/libero_evaluation/validate_delivery.py /data/libero130/evaluation/delivery
```

打包器逐项核对原始 BDDL、初态资产、连续状态和动作，转码并完整解码视频，按相同 16 类统计，生成内嵌全部视频的离线 HTML。源代码、运行协议和校验结果可追溯。大输出和模型不提交 Git。

此版本的打包与展示以本轮提交和日期为边界，后续策略迭代应建立新的实验协议和报告版本。每任务一个初态是工程回归，不是官方多初态基准成绩。使用仿真真值的 teacher 条件不等价于仅 RGB 的学生策略；阶段关键点来自手写 code policy，不能直接视作 VLM 决策标签或已审核的 SFT 数据。两个堆叠后整体搬运任务另外记录语言顺序问题。

## 新策略的针对性复验

`run_subset.py` 接收明确的任务键列表，使用同样的驱动、初态和任务预算。输出逐项视频、原始结果、实际配置和源码来源。它不重试失败，也不将历史成功计入新版本。

```bash
python tools/libero_evaluation/run_subset.py \
  --source /data/frozen-new/grasp-pipeline --paths /data/paths.local.json \
  --output /data/new-subset --revision YOUR_FROZEN_COMMIT \
  --gpu-slots 0 0 1 1 2 2 3 3 \
  --keys libero_goal_open_the_middle_drawer_of_the_cabinet
```

上面的 130 项打包器仍服务于冻结基线；新版部分复验另建报告，不能混用该基线的提交标签和成功总数。新策略修改说明见 [环境筛选与机构位姿迭代](../../docs/GRASP_ENVIRONMENT_SCREEN.md)。

本轮部分复验的打包与独立审核方式：

```bash
python tools/libero_evaluation/package_subset.py /data/new-subset \
  --baseline /data/libero130/evaluation/delivery --source-archive /data/libero-frozen.tar.gz
python tools/libero_evaluation/validate_subset.py /data/new-subset/delivery
```

源码归档须由对应冻结提交的 `git archive` 生成，并保留提交标记。实验目录的上一层还需包含 `FROZEN_SOURCE.json`、`REGRESSION_FROZEN.xml`、`EXTERNAL_SOURCE_BEFORE.json`、`EXTERNAL_PRECHECK.json` 和 `FINAL_LAUNCH.json`。本轮工具校验146项回归记录，适用于本次修订；新修订应同步更新协议和验证要求。

`finalize_subset.py` 可等待全部明确列出的任务结束，再调用上述打包与审核、生成ZIP和交付校验清单；它不会新增任务或重试失败。最终结果见 [39 项同版复验](../../docs/LIBERO_ENVSCREEN_20261113.md)。
