# 独立 RoboCasa 仿真环境

这是开发中的接入环境；当前验证范围见[接入状态](../../docs/robocasa/IMPLEMENTATION_STATUS.md)。不要将这些依赖安装到 LIBERO、GraspGen 或 cuRobo 原有环境中。

实际验证机器使用 Linux、NVIDIA RTX 4090 与 Python 3.11.10。新仿真环境单独加载 robosuite 和 RoboCasa；两个模型 worker 继续使用各自解释器。源码固定为：

| 上游 | commit |
| --- | --- |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| RoboCasa | `3443cdf3dd20f9ae8ed41adba442059b506891d0` |

在有足够空间的数据盘创建全新虚拟环境与源码 checkout。先安装 [requirements-simulation.txt](requirements-simulation.txt)，再在该环境安装两个固定源码目录：

```bash
python -m pip install -r deploy/robocasa/requirements-simulation.txt
python -m pip install --no-deps -e /data/robosuite
python -m pip install --no-deps -e /data/robocasa
```

这里列出已测环境中的主要依赖；它不是上游训练环境的完整安装脚本。PandaOmron 使用原生 OSC composite controller，未声明 GR1、mink 全身控制或 RPent RLDX 的支持。新机器仍需通过原生 probe、FK 和物理控制验收。

将 `configs/robocasa/paths.example.json` 复制为自己的 `paths.local.json`，填写七项绝对路径。`robosuite_assets` 必须指向新固定 robosuite 的 `robosuite/models/assets`，不能沿用 LIBERO 1.4.1 的资产。

资产下载用 `tools/robocasa/download_assets.py`，固定两个公共数据仓库的 revision 并校验 SHA256。数据盘需同时容纳压缩包、解压文件、临时文件和轨迹；安装器逐包检查可用空间。不要用“目录已存在”作为资产完整性的依据。

若系统盘空间不足，设置数据盘上的 `TMPDIR`、`PYTHONPYCACHEPREFIX`、`NUMBA_CACHE_DIR`；程序也会为 worker 配置数据盘上的 Warp / PyTorch 缓存。路径不使用机器通用的 HOME 变量替代。

几何依赖必须在 EGL 相机创建前加载，运行入口已经包含这一修复。自行编写诊断脚本时调用 `prepare_render_dependencies()`，否则已测版本组合可能在首次渲染后出现损坏画面。

资源下载、配置预览和测试不等于物理任务成功；只有对应回合的原生目标、完整阶段、真实动作、版本指纹和有效录像才能进入任务结果。
