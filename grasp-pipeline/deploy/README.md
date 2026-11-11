# 部署与启动检查

本目录交付的是源码 checkout、冻结参考、接口文档和检查工具。实际运行环境仍为 LIBERO / MuJoCo / 单 Panda。三个 Python 环境、上游源码、权重、资产、CUDA 和 EGL 需要另行准备。其他 benchmark 的接入见[接入实施指南](../docs/integration/README.md)。

## 1. 解压后先核对代码

在交付包最外层执行（此层包含 `grasp-pipeline/` 和 `MANIFEST.json`）：

```bash
python3 grasp-pipeline/tools/build_handoff.py --verify .
cd grasp-pipeline
python3 tools/verify_preservation.py
python3 tools/check_integration_docs.py
python3 run.py list
```

这些命令只需要 Python 标准库，不安装软件、不创建环境、不使用 GPU。包内附带 `../pipeline-organization/baseline/v10/`，因此版本核对不依赖原工作区。校验清单校对内容完整性，不是数字签名。

开发主机可用 Python ≥3.10 查看配置和文档；完整实验的已记录环境是 Python 3.11.10。`pyproject.toml` 中的最低版本不能当作所有第三方依赖的兼容承诺。

## 2. 准备外部源码、资产与模型

先在独立的部署目录准备下列资源，记录 commit 与本地修改。下面是原服务器在本次交付时核对的版本，详情见 [SOURCES.json](reference/SOURCES.json)。

| 资源 | 已核对的版本或要求 | 负责提供什么 |
| --- | --- | --- |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | `libero/libero/` 源码、BDDL、`init_files/`、`assets/` |
| graspgen-curobo-collect | `b69b3133badefd80e7c13c57119a487c97fec614` | 本项目使用的 `third_party/GraspGen/` 源码布局 |
| GraspGen | 上述 fork 内的嵌入源码树 | 模型代码；不能把 fork 的 git HEAD 误当成独立 GraspGen checkout 的 commit |
| cuRobo | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，模块版本 0.7.8 | CUDA 扩展、规划模块及自身资产 |
| robosuite | 1.4.1 | Panda / PandaGripper 的原始 MJCF、mesh 与控制器 |
| Panda 模型 | 官方生成器、判别器及其 YAML | `franka_panda`、2048 点；两个权重与配置实际引用对应 |

上游安装和资源说明：[LIBERO 固定版本 README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/README.md)、[GraspGen 官方仓库](https://github.com/NVlabs/GraspGen)、[cuRobo v0.7.8](https://github.com/NVlabs/curobo/tree/v0.7.8)、[使用的 fork](https://github.com/juchuanqi/graspgen-curobo-collect)。fork 的获取权限由接入方解决；本包没有重分发这些仓库、模型或资产。

官方文档的默认 Python、Torch 版本可能与本实验不同。例如原 LIBERO README 面向其原始训练环境，不能把它的旧依赖清单直接装进本实验的三个环境。以固定源码、下面的角色版本、实测记录与检查结果为依据，解决每个环境自己的依赖。不要安装最新 main 后仍宣称与 v10 环境一致。

Panda 权重的基线 SHA256：

```text
generator     0597583b89b322d42ceb4e596967d6ed68d1b56cba4039895909ccd5bdc66eff
discriminator e47d703c63b54c2d11fbc1effd43898f251b4147250888541e3b16e9c0d19e1c
```

配置加载规则与原 GraspGen 保持一致：`eval.checkpoint` 和 `discriminator.checkpoint` 相对 YAML 所在目录解析，绝对路径则保持绝对路径。检查工具读取实际 YAML，不凭固定文件名猜测模型位置；只读取权重内容计算指纹，不反序列化权重。

本管线运行不需要下载全部人类演示训练数据；任务 BDDL、对应初态和场景资产必须完整。目录存在不代表每个任务资产都有效，完整性还要通过环境诊断和实际任务核验。

## 3. 准备三个独立运行环境

建议分别建立仿真、模型和规划环境，避免混装冲突。在所选 Python 3.11 环境中，按照固定上游版本安装其所需依赖。依赖角色如下；这张表和[核心版本记录](reference/dependencies.v10.json)都不是完整的可安装锁文件。

| 角色 | 核心实测版本 | 额外要求 |
| --- | --- | --- |
| 仿真与控制 | NumPy 1.26.4、SciPy 1.16.2、Torch 2.6.0+cu124、MuJoCo 3.3.7、robosuite 1.4.1、OpenCV 4.11.0 | LIBERO 及其依赖、图像/视频编码、EGL |
| GraspGen | NumPy 1.26.4、SciPy 1.16.2、Torch 2.6.0+cu124、OmegaConf 2.3.1 | 固定 fork 的 GraspGen 全部依赖和其 CUDA 扩展 |
| cuRobo | NumPy 1.26.4、SciPy 1.17.1、Torch 2.6.0+cu124、cuRobo 0.7.8 | 构建工具链、匹配的 CUDA 编译环境和机器人资产；原 planner 还要能定位 robosuite 资产 |
| 离线检查，可独立于三个 actor 环境 | pytest、[requirements-check.txt](../contracts/requirements-check.txt) | Schema 与 CPU 回归；测试中的几何部分另需仿真 CPU 依赖 |

[RUNTIME_INVENTORY.json](reference/RUNTIME_INVENTORY.json) 保存本轮在原服务器观察到的 Python、模块位置、已安装 distribution 名称和版本。它帮助排查差异，但没有 wheel、系统库、完整依赖来源或 CUDA 驱动，不能直接当作 `pip install -r` 的文件。此次未在一台全新机器上从零安装三个环境。

本包可以直接从 checkout 的 `run.py` 启动，无须为了加载本包而复制源码进三个环境。如需 editable 安装，在目标解释器中执行 `python -m pip install --no-deps -e .`；它只安装本包入口，不安装第三方运行依赖。配置与文档仍依赖这个 checkout，不提供脱离 checkout 的完整 wheel。

完整运行要求 Linux NVIDIA GPU 和可工作的驱动、CUDA、EGL。原实验使用 RTX 4090；本阶段没有测定最低显存或其他 GPU 的性能。`torch` 导入成功、`nvidia-smi` 正常都不等于 CUDA 内核和 EGL 已通过验证。

## 4. 填写六个部署路径

首次部署时从 [paths.libero.example.json](../docs/integration/templates/paths.libero.example.json) 复制一份到 `configs/runtime/paths.local.json`，将六个占位路径全部改为当前机器的绝对路径：

| 键 | 指向什么 |
| --- | --- |
| `libero_repo` | 包含 `libero/libero/__init__.py` 的 LIBERO checkout 根目录 |
| `grasp_repo` | 包含 `third_party/GraspGen/grasp_gen/` 的 fork 根目录 |
| `graspgen_python` | 模型环境可执行的 Python |
| `graspgen_config` | 实际 Panda YAML，内部引用真实权重 |
| `planner_python` | 规划环境 Python，或最终执行该 Python 的可执行 wrapper |
| `simulation_python` | 仿真与控制环境 Python |

新路径文件只接受这六项。不要把抬升高度、搜索预算等策略参数混入；也不要用其他 benchmark 的设计单充当此路径文件。

如果规划环境需要额外的 `CUDA_HOME`、`PATH`、`TORCH_EXTENSIONS_DIR` 或 GPU 架构设置，用本机 wrapper 设置后 `exec /absolute/planner/bin/python "$@"`。不要照抄原服务器的 `/opt` 别名、缓存位置或 RTX 4090 架构值。wrapper 必须完整转发参数，以下检查会通过 wrapper 执行。

## 5. 路径、CPU 导入和权重检查

```bash
python3 tools/check_deployment.py --paths configs/runtime/paths.local.json --report checks/paths.json
python3 tools/check_deployment.py --paths configs/runtime/paths.local.json --probe-runtime --hash-weights --report checks/deployment.json
```

第二条依次调用三个解释器，隐藏 GPU，导入记录的核心模块、比对版本、读取 Panda 配置、核对实际两个权重。退出码 0 表示所请求的检查通过；缺路径、模块导入失败、版本不符或权重不同均返回 1，非法选项返回 2。报告中有具体失败项。默认每角色 60 秒，可通过 `--timeout` 调整检查超时，不改变策略参数。

`--probe-runtime` 不导入完整 GraspGen 网络、不构造 MotionGen、不创建仿真、不做 CUDA/EGL 测试。它和 worker 的 `--help` 都属于启动前检查，不是模型加载验收。检测到的版本变化应先查清原因；若是有意升级，应单独记录部署版本和验证，不能覆盖旧基线声明一致。

## 6. 环境诊断与完整任务

先让配置预览使用新路径：

```bash
python3 run.py run libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket --gpu 0 --init 0 --tag deploy-check --paths configs/runtime/paths.local.json --dry-run
```

去掉 `--dry-run` 会真正加载模型、规划和执行任务，需要空闲 GPU；单任务三个进程使用同一卡。下面命令形态用于部署方的后续验收，本阶段没有执行新的任务轨迹：

```bash
python3 run.py run libero_object_pick_up_the_orange_juice_and_place_it_in_the_basket --gpu 0 --init 0 --tag deploy-check --paths configs/runtime/paths.local.json
```

已有输出目录会被拒绝复用。换 tag 或设置新的 `--output-root`，不要删除历史结果来伪装成新回合。输出默认为 `runs/<tag>/<task_key>/`。

环境与坐标诊断可以通过旧单任务 `legacy doctor` 执行，但应使用所选仿真解释器启动 `run.py`，并明确给它配置和新输出目录。它只检查自身场景与几何，不是完整 130 任务配置；不应把 doctor 通过记为抓放成功。具体分层验收见 [G0–G8](../docs/integration/DEBUG_VALIDATE.md)。

最终以根 `RESULT.json` 的真实动作、`pipeline_success` 和 `official_goal_at_end`、阶段记录和视频共同核验。指令顺序、稳定性和训练数据完整性还要另外审核。历史 v10 的 88 / 86 / 85 三种统计口径见[版本与验证](../docs/RELEASE_VALIDATION.md)，本轮没有产生新成功率。

## 常见问题

| 症状 | 优先检查 |
| --- | --- |
| 只会显示配置，运行时缺模块 | 仿真解释器选错；系统 Python 能预览不表示 actor 依赖齐全 |
| 模型配置存在但找不到权重 | 实际 YAML 的两个 checkpoint 引用及相对目录，不只检查默认文件名 |
| 模型侧 CUDA 扩展失败 | 固定 GraspGen 依赖、Torch/CUDA ABI、扩展构建；CPU 检查不覆盖这些 |
| 规划器启动失败或资产缺失 | wrapper 转发、cuRobo import、工具链及 robosuite MJCF/mesh 路径 |
| 黑屏或 EGL 错误 | NVIDIA/EGL/容器设备映射与渲染设备；不要改抓放参数来处理渲染问题 |
| 不同机器人或 benchmark 不能直接运行 | 当前运行链仍依赖 Panda/LIBERO；按文件改动清单实现并接通后端 |

