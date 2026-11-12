# RoboCasa 配置

| 文件 | 用途 |
| --- | --- |
| `paths.example.json` | 七项机器路径模板，复制成被 Git 忽略的 `paths.local.json` 后使用 |
| `policy.json` | r10 共享策略与未开放能力开关；有效 v10 规则由配置加载器从 LIBERO 基线读取 |
| `tasks.json` | 12 项内置语义绑定，不保存逐任务/初态的世界坐标或轨迹 |

配置加载器位于 `src/graspgen_libero_bridge/adapters/robocasa/config.py`。65 项覆盖的额外语义清单位于 `tools/robocasa/atomic65.py`，与内置 CLI 目录分开；43 项没有动作策略，明确标为场景探针。

`native` 使用官方 horizon；`libero-v10-teacher` 显式选择原 v10 的 5000 步采集预算。两者都保留原生任务成功函数，扩预算成功不能计为原生预算成功。

`articulation_enabled`、`base_motion_enabled`、`composite_enabled` 和 `carried_contents_enabled` 当前为 false。启用前需要完整的任务/机构绑定与物理验证，不能只修改布尔值后声称能力可用。

保留的 r7 候选和搬运参数不控制已继承的 v10 主搜索。实际来源见[迁移记录](../../docs/robocasa/V10_MIGRATION.md)，不要仅根据字段名称判断是否生效。
