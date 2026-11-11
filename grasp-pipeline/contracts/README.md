# 第一版接入约定文件

规范入口：[接口说明](../docs/interfaces/README.md)。本目录保存数据 Schema、当前实现能力要求和离线参考数据，不是模型权重或运行参数目录。

- `pipeline-v1.schema.json`：14 类公开数据包和 30 个总结构定义。
- `profiles/libero_v10_teacher.json`：当前 robosuite Panda、Panda 夹爪及 teacher 管线需要的能力，不修改原参数。
- `LIBERO_V10_BINDINGS.json`：九类接口的 11 个现有源码锚点和剩余绑定工作。
- `examples/libero_v10_reference.json`：真实旧记录转换的参考，有缺失项，不能直接执行。
- `examples/libero_v10_outcomes.json`：130 条原结果的规范化表示，保持 88/86/85 统计口径。
- `requirements-check.txt`：离线校验所需的可选开发依赖，不是 actor 依赖清单。

Schema 和样例可用对应的 `tools/build_contract_schema.py`、`tools/build_libero_contract_map.py`、`tools/export_contract_reference.py` 重建。导出样例需要已有 delivery 和 NumPy；校验器需要 jsonschema。所有操作均无需模型推理或 GPU。
