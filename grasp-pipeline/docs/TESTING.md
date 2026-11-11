# 测试与验证范围

## 本地只读核验

```bash
python3 tools/verify_preservation.py
```

核对内容包括：

- 39 个冻结 Python 文件全部有对应位置。
- 每个文件仅发生迁移记录中列出的机械替换；未记录的代码变化会失败。
- 所有新内部导入模块都存在。
- 原 130 任务清单和旧单任务示例逐字节一致。
- 从原驱动独立提取的 15 个静态覆盖值与新 JSON 相同。
- 130 任务 × 4 个 GPU 编号 × 2 个初态绑定，共 1040 组有效配置完全一致。

初态 1 在这里只用于检查参数绑定，没有执行该初态任务。已有测评仍全部是初态 0。

`migration/PRESERVATION_CHECK.json` 保存本地核验结果。需要另一个基线位置时使用 `--baseline /path/to/v10`。

## 原测试和新增入口测试

在具有原仿真 CPU 依赖的解释器中，从本目录运行：

```bash
PYTHONPATH=src python -m pytest -q tests
```

原有测试检查坐标转换、点云、Panda FK、真实 MuJoCo 局部 IK、碰撞拒绝、预演状态恢复、控制超时、限位、夹爪交接、任务编排、搬运放置搜索和真实本机 RPC 子进程。

新增的四项配置与入口测试还可单独运行，不需要 NumPy、MuJoCo 或 GPU：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_runtime_config.py -v
```

它们检查部署配置不能覆盖策略参数、迁移路径只改变五个路径字段、dry-run 不启动仿真，以及源码指纹覆盖嵌套模块。

## 第二阶段服务器验证

在原四卡服务器的独立目录中，使用最终 v10 的 Python 3.11.10 环境，对上传的原版副本和整理版副本分别运行测试。所有测试子进程的 GPU 均隐藏，未进行模型推理或完整任务执行。

原版 86 项测试全部通过，整理版 90 项全部通过，其中包括全部原测试和新增四项测试，无失败、无跳过，原测试身份与结果一致。模型和规划工作进程的 `--help` 入口检查也通过。

本地 71 个代码与配置文件和通过测试的服务器副本哈希一致。完整记录见 [CPU_VALIDATION.json](../migration/CPU_VALIDATION.json)，原日志和 JUnit 位于 [migration/cpu](../migration/cpu/)，源码指纹见 [TESTED_SOURCE_SHA256.json](../migration/TESTED_SOURCE_SHA256.json)。

当前没有重新执行 130 任务，不能把 CPU 回归通过视为已测得整理版新成功率。

## 第三阶段与交付验证

第三阶段验证了全部 113 项 CPU 测试，增加了接口与真实旧记录检查；日志、Schema 验收与指纹位于 [migration/phase3](../migration/phase3/)。这是第三阶段的历史验证记录。

第五阶段新增部署失败分类和交付包防篡改检查，不改 actor 代码。新增八项工具测试可以在标准库环境单独执行：

```bash
python3 -m unittest discover -s tests -p test_deployment_tools.py -v
```

它们检查缺依赖/超时不能被误报为通过、路径文件不能夹带策略参数、路径通过不能冒称 GPU 验收，以及交付文件变动/多文件/重复输出的拒绝处理。第五阶段实际运行证据见 [migration/phase5/VALIDATION.json](../migration/phase5/VALIDATION.json)。部署检查和完整 GPU 任务验收的边界见[部署说明](../deploy/README.md)。
