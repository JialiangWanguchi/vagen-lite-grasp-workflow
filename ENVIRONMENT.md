# 环境迁移清单

训练入口已参数化；环境仍需在目标 GPU 上安装并验收。本机不下载模型或数据。

## 已验证环境

Python 3.10.12，Linux，CUDA 驱动 550.90.07；RTX 3060 12GB。

| 组件 | 固定版本 |
|---|---|
| PyTorch | 2.8.0+cu128 |
| vLLM | 0.11.0 |
| Transformers | 4.57.1 |
| PEFT | 0.17.1 |
| FlashAttention | 2.8.3 |
| Ray | 2.49.2 |
| TensorDict | 0.10.0 |
| VAGEN-Lite | 04bf4bd13bd93688d5cd66331745190486fd14d1 |
| VERL | 3fe0a29975e1b02ae2bd1dec249f7807dd7966f5 |

发布快照见 [requirements-tested.txt](requirements-tested.txt)，已去除机器特定 editable 安装路径，VAGEN/VERL/compat 需按 README 从源码安装；它不是跨平台 lockfile。历史代码指纹见 [environment.json](docs/results/2026-09-05/environment.json)。新运行仍生成 `reports/pip_freeze.txt` 和 `reports/environment.json`，请勿直接发布未经审查的本机快照。

## 新服务器准备顺序

1. 在目标服务器创建 Python 3.10 虚拟环境，默认名为工程下 `venv`，或者设置 `GRASP_VENV`。检查 GPU 驱动、系统内存、磁盘、CUDA 编译工具与 Python 开发头文件。不要直接复制旧 venv 的绝对路径文件。
2. 获取 VAGEN 的 `vagen-lite` 分支并固定到上述提交，VERL 子模块固定到上述提交。若使用不同目录名，配置 `VAGEN_DIR`、`VERL_DIR`。
3. 根据实际版本快照安装依赖；先确保 PyTorch、vLLM、Transformers 的版本组合一致，再安装其他依赖和两个源码工程的 editable 包。
4. 安装本工程补丁包：`python -m pip install --no-deps -e ./compat`。`runtime.sh` 会启用 `grasp_qwen3_vl` 插件。
5. FlashAttention 必须与 Python、PyTorch、CUDA、C++ ABI 匹配。本次使用官方 cp310 / torch2.8 / cu12 / cxx11abiTRUE 预编译包。不同平台不可盲用此 wheel，必要时从源代码构建。
6. 在新服务器下载或从旧服务器传输模型与数据，配置 JSON 中路径。运行配置单元测试、实际数据预检和短步数训练，确认权重更新、导出和重载通过。

旧的 `download_*.py/sh`、`install_headers.sh`、`install_verl.sh`、`prepare_remote.py` 与 `prepare_model_view.py` 是本次服务器部署的过程记录，包含这台机器和本次数据的固定值；它们不属于通用训练运行入口。迁移不应直接运行这些旧脚本。

密集 Qwen3-VL 模型可直接在 profile 中指定其原始模型目录，`vision` 配置显式控制处理器参数，不强制创建 `-224` 目录。是否能在目标分辨率和 GPU 上运行，以 `preflight.py` 与短跑为准。

大规模运行前还需重新设计病例隔离的划分、训练预算和正式评测。本次包不承诺自动租卡、多节点调度、任意模型架构或任意版本升级兼容。
