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

模型权重和上游源文件的 `EXTERNAL_SOURCE_BEFORE.json`、部署检查及回归记录放在实验根目录（evaluation 的上一层）。只在全部 130 项结束后执行：

```bash
python tools/libero_evaluation/package_results.py /data/libero130/evaluation
python tools/libero_evaluation/validate_delivery.py /data/libero130/evaluation/delivery
```

打包器逐项核对原始 BDDL、初态资产、连续状态和动作，转码并完整解码视频，按相同 16 类统计，生成内嵌全部视频的离线 HTML。源代码、运行协议和校验结果可追溯。大输出和模型不提交 Git。

此版本的打包与展示以本轮提交和日期为边界，后续策略迭代应建立新的实验协议和报告版本。每任务一个初态是工程回归，不是官方多初态基准成绩。使用仿真真值的 teacher 条件不等价于仅 RGB 的学生策略；阶段关键点来自手写 code policy，不能直接视作 VLM 决策标签或已审核的 SFT 数据。两个堆叠后整体搬运任务另外记录语言顺序问题。
