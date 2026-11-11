# 配置样例与启动方式

## 四种样例各自有什么用途

| 文件 | 是否由当前运行入口消费 | 用途 |
| --- | --- | --- |
| [paths.libero.example.json](templates/paths.libero.example.json) | 格式可由当前 `--paths` 加载，需替换为真实绝对路径 | 仅演示当前 LIBERO 的部署路径，不是其他 benchmark 配置 |
| [integration_spec.example.json](templates/integration_spec.example.json) | 否 | 接入设计单，记录待定项、版本、机器人、来源、后端和责任人 |
| [task_spec.example.json](templates/task_spec.example.json) | 否；可按 TaskSpec Schema 校验 | 说明“杯子放入容器”的目标数据结构；不是已注册任务 |
| [capabilities.example.json](templates/capabilities.example.json) | 否；可按 Capabilities Schema 校验 | 未实现提供方的诚实初始声明，能力全部为 false |

设计单没有被 `runtime/config.py` 读取。新环境部署键的名称由接入方实现并文档化；不要把新环境路径放进 `libero_repo`，也不要向六路径文件添加 `benchmark_repo` 后期望旧加载器自动识别。

## 现有命令能验证什么

以下命令从代码根目录运行，不需要启动模型或环境：

```bash
python3 run.py --help
python3 run.py list --json
python3 run.py config libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket --gpu 0 --init 0
python3 run.py run libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket --gpu 0 --init 0 --tag integration-preview --dry-run
```

它们验证当前 LIBERO 的配置和命令接线；`--dry-run` 不调用仿真，不能证明新 benchmark 已接通。去掉它才会启动当前配置的仿真解释器和 LIBERO 任务。

`python3 run.py legacy doctor --config configs/examples/single_task.v10.json` 是已有的 LIBERO 环境/几何诊断入口，需要对应仿真依赖和渲染条件。它不验证完整 GraspGen/cuRobo 链，也不能检查其他 benchmark。

## 新环境需要单独实现的启动设置

接入方的新入口应显式绑定：环境提供方、任务配置、机器人/夹爪模型、观测模式、原生动作规格、规划及局部 IK 后端、成功判定、输出目录和预算。把这些选择写入实际运行产物，不能只保存在启动脚本的隐含默认值里。

建议分开保存环境部署路径、机器人标定、策略参数、任务/初态清单和评测预算。位置/旋转误差容差、抬升高度、接近距离、放置候选、限位余量等迁移相关参数应说明来源。保持 v10 的对照实验使用原值；如为了新机器人或任务改动，记录新配置版本，不能继续称为原参数。

同一张卡可以承载多个独立进程，但并发量取决于模型、规划、渲染、CPU 和缓存；显存未占满不代表能无限加并发。先测单任务峰值和耗时，再决定调度，不把原四卡实验槽位照抄为硬件要求。

## 校验样例和自己的数据

开发解释器需要可选的 `jsonschema` 依赖，具体版本见 `contracts/requirements-check.txt`；它不要求安装到仿真、GraspGen 和规划三个环境里。

下面代码是真实可执行的 Schema 示例，只检查已有的示例文件：

```bash
PYTHONPATH=src python - <<'PY'
import json
from graspgen_libero_bridge.contracts.validation import validate_payload
with open('docs/integration/templates/task_spec.example.json', encoding='utf-8') as f:
    validate_payload(json.load(f))
with open('docs/integration/templates/capabilities.example.json', encoding='utf-8') as f:
    validate_payload(json.load(f))
print('Sample structures valid; no environment or task was run.')
PY
```

接入方已有自己的 Observation、TaskSpec、SceneSnapshot 等公开数据包后，可用同一个 `validate_payload` 检查结构和部分语义。RobotState 等嵌套结构放在对应公开包里校验。任务能否由某套后端执行，另做本环境的 `check_requirements`、资源解析、FK、预检和真实运行测试。

当前 `readiness(bundle)` 的 profile 固定为 LIBERO teacher，`check_contracts.py` 固定验证仓库内历史样例。二者不接受任意新 benchmark 配置，也不会因为新写了 profile JSON 就自动切换。

本轮文档的维护检查：

```bash
python3 tools/check_integration_docs.py
```

该脚本核对文档、本次现有源码、示例格式和轻量 CLI；它不是新环境启动器。具备 jsonschema 时可加 `--schema` 检查两个公开示例的数据结构。单独拷贝本代码目录时不要求 RPent checkout；若没有本地 RPent，报告会注明没有进行该项本地源码核验，公开的固定 commit 链接仍可查阅。
