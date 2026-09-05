# VAGEN-Lite · Qwen3-VL · 三组可迁移训练工作流

面向 **2D 图片 + 文字、单轮交互（`max_turns=1`）** 的真实数据小规模流程验证：

1. 单独 SFT（LoRA）；
2. 单独 RL（GRPO + LoRA）；
3. 独立 SFT 后继续 GRPO。

三组代码和脚本独立，配置、数据契约、图像处理、奖励与测试共用。目标是先在小 GPU 上打通流程，随后在新服务器调整模型、数据路径和训练参数，继续使用这些入口。

**已验证：** 2026-09-05，单张 RTX 3060 12 GiB，Qwen3-VL-2B-Instruct，三组真实优化器更新、LoRA 导出、组合组权重交接及 vLLM 重载测试全部完成。**这不是模型效果达标或多卡性能验收。** 本仓库不包含数据、图片、模型、逐条预测、原始运行日志或聊天记录。

## 1. 框架口径

vLLM 负责生成/推理，不承担反向传播。这里统一的是同一套训练与推理技术栈，而不是声称“用 vLLM 做 SFT 梯度训练”：

| 组别 | 梯度训练 | 训练采样 | 共同测试 |
|---|---|---|---|
| SFT-only | VERL / FSDP + LoRA | 不需要 rollout | vLLM |
| GRPO-only | VAGEN-Lite → VERL / FSDP + LoRA | vLLM | vLLM |
| SFT→GRPO | 同上，独立 SFT 后加载其 LoRA 做 GRPO | vLLM（RL 阶段） | vLLM |

Transformers 提供模型类和处理器；没有另起一套 Transformers Trainer 来训练 SFT。使用 **VAGEN 的 `vagen-lite` 分支固定提交**，不依赖默认分支的后续变化。无 3D 环境、无外部 Judge API，奖励直接对照数据的真实标签。

## 2. 实验结果速览

完整分析见 [WORKFLOW_REPORT.md](WORKFLOW_REPORT.md)，机器可读汇总见 [final_results.json](docs/results/2026-09-05/final_results.json)。

200 条原始样本：A2、P1 各 100 条；固定划分为训练 140、验证 20、测试 40。本次每个训练阶段仅 **4 步**，测试固定使用 A2/P1 各 4 条，共 8 条；不是用全部数据充分训练。

| 模型/组别 | A2 正确 | P1 正确 | 总正确 / 8 | 严格准确率 | JSON 合法率 | 输出截断数 |
|---|---:|---:|---:|---:|---:|---:|
| 原始基座 | 0/4 | 0/4 | 0/8 | 0% | 37.5% | 1 |
| SFT-only | 0/4 | 0/4 | 0/8 | 0% | 62.5% | 3 |
| GRPO-only | 0/4 | 0/4 | 0/8 | 0% | 12.5% | 1 |
| SFT→GRPO | 1/4 | 0/4 | 1/8 | 12.5% | 62.5% | 2 |

组合组相对基座为 **+12.5 个百分点**；因为基座为 0，相对百分比增幅未定义。只有一个种子、8 条测试样本，不能据此宣布组合方法优于其他方法，也不报告虚构的多种子均值、标准差或显著性。

流程完成的证据不是“loss 打印出来了”或“程序没报错”：

- 三个最终适配器各有 392 个张量、8,716,288 个参数；全部有限值，未包含视觉层 LoRA。
- SFT 4 步 loss 从 2.07406 到 0.970598；不同批次的首尾值不构成收敛证明。
- GRPO-only 在第 2 步出现非零策略梯度；组合组的 GRPO 在前 2 步出现非零策略梯度。其余部分步骤奖励全同、梯度为 0，不能把执行步数都算成有效学习。
- 组合组 GRPO 权重与其独立 SFT 阶段的参数差 L2 为 **0.06615864855165701**，证明实际接续并更新了权重。
- 七个训练/测试阶段退出码均为 0，RL 物理显存采样峰值 11,346 MiB。原始外层 `workflow_suite` 因报告时间解析兼容问题退出 1；修复后单独 `report_finalization` 退出 0，**没有重新训练或改写预测来掩盖该失败**。

当前准确率和格式表现较弱，说明输出约束、训练预算、模型容量及视觉分辨率需要后续研究；本次实验无法区分它们各自的影响。相同图片 SHA 不跨集合，但不保证病例或邻近帧隔离，不能声称病例泛化。独立实验审计未完成，详见 [审计状态](EXPERIMENT_AUDIT.md)。

## 3. 代码导航

| 文件 | 用途 |
|---|---|
| `train_sft.py` / `run_sft.sh` | 独立视觉 SFT / LoRA |
| `train_grpo.py` / `run_grpo.sh` | VAGEN-Lite 单轮 GRPO，可显式传入 adapter |
| `train_sft_grpo.py` / `run_sft_grpo.sh` | 从基座独立 SFT，再将导出的 LoRA 传给 GRPO |
| `evaluate_vllm.py` | 基座及三组共同的 vLLM 贪心评测 |
| `experiment_config.py` / `profiles/` | 默认参数、深度合并、校验和硬件规模模板 |
| `task_contract.py` | 提示词、真实标签、SFT 目标与严格 JSON 奖励 |
| `grasp_common.py` / `grasp_sft_dataset.py` | 全组共用图片加载；SFT 多模态数据适配 |
| `grasp_env.py` / `grasp_seed.py` | 一次回答即终止的环境与训练种子 |
| `compat/` | 固定 vLLM 版本的 Qwen3-VL 视觉前缀兼容插件 |
| `preflight.py` / `audit_adapter.py` | 实际图片/token 预检及适配器验收 |
| `run_all.sh` / `run_logged.sh` | 完整流程与独立日志、退出码、显存采样 |
| `status.py` / `finalize_results.sh` | 进度查询；只重建报告而不重跑训练 |
| `summarize_results.py` / `write_workflow_report.py` | 复算共同指标并生成报告 |
| `docs/results/2026-09-05/` | 本次运行的脱敏聚合结果、环境和来源指纹 |

## 4. 远端安装与准备

仅在 Linux GPU 服务器执行训练、下载模型和处理数据。本地 Windows 可用于查看代码和 CPU 单元测试。

固定环境：Python 3.10.12、PyTorch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.57.1、PEFT 0.17.1、FlashAttention 2.8.3、Ray 2.49.2、TensorDict 0.10.0。详细说明见 [ENVIRONMENT.md](ENVIRONMENT.md)。

```bash
# 在新服务器选择自己的工作目录；私有仓库需要先配置 GitHub 访问权限
git clone https://github.com/JialiangWanguchi/vagen-lite-grasp-workflow.git
cd vagen-lite-grasp-workflow
export GRASP_ROOT="$PWD"
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

git clone --branch vagen-lite https://github.com/mll-lab-nu/VAGEN.git VAGEN-vagen-lite
git -C VAGEN-vagen-lite checkout 04bf4bd13bd93688d5cd66331745190486fd14d1
git -C VAGEN-vagen-lite submodule update --init --recursive
export VAGEN_DIR="$PWD/VAGEN-vagen-lite"
export VERL_DIR="$VAGEN_DIR/verl"
test "$(git -C "$VERL_DIR" rev-parse HEAD)" = 3fe0a29975e1b02ae2bd1dec249f7807dd7966f5

# 先确认目标 GPU/驱动支持该 CUDA 组合，且有匹配的系统构建依赖
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# 安装匹配的 FlashAttention wheel，或准备 nvcc / Python.h / 编译器后构建。
# requirements-tested.txt 为本次已安装环境的版本快照，不是跨平台 lockfile。
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps --no-build-isolation -e "$VERL_DIR" -e "$VAGEN_DIR"
python -m pip install --no-deps -e ./compat
python -m pip check
```

安装配方由已完成环境整理而来，本次发布没有在全新机器重新安装全部依赖；不要把它当成已验证的任意服务器一键安装器。FlashAttention/驱动/ABI 最容易出现差异。若使用其他目录，每次启动前保留 `VAGEN_DIR`、`VERL_DIR`、`GRASP_VENV` 设置。

在服务器获取 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，本次固定 revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。例如下载到默认模型路径：

```bash
# 下载时不要先 source runtime.sh，它会开启 Hugging Face 离线模式
hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir models/Qwen3-VL-2B-Instruct-224
```

目录后缀 `-224` 只是约定名称，图像处理由 profile 的 `vision` 字段控制；也可直接配置已有原始模型目录。文本版 Qwen3 不接受图片，这里使用的是 Qwen3 系列的 **VL dense** 模型。

准备 `prepared/train.jsonl`、`val.jsonl`、`test.jsonl`，以及 `dataset/data/GraSP/` 下的图片，或在 profile 中配置已有路径。数据不在 GitHub；字段与拆分要求见 [数据契约](docs/DATA_CONTRACT.md)。当前仓库不附带针对新数据自动划分的通用脚本。

## 5. 先预检，再运行三个独立实验

```bash
export GRASP_CONFIG="$PWD/profiles/workflow_3060.json"
export CUDA_VISIBLE_DEVICES=0
source runtime.sh
python -m unittest -v test_experiment_config.py test_task_contract.py
python task_contract.py
python preflight.py

# 三组训练：使用不同且全新的输出目录
bash run_sft.sh --output runs/my_sft
bash run_grpo.sh --output runs/my_grpo
bash run_sft_grpo.sh --output runs/my_sft_grpo
```

预检读取真实图片并计算处理器实际 token 长度，检查 dense 架构、标签、两任务、图像数、长度上限与可用训练批次；它不是病例泄漏检查器。当前 SFT 要求每卡 microbatch=1，有效 batch 通过梯度累积增加。

或者在**全新工作/输出空间**一键执行基座测试 + 三组训练 + 权重验收 + 共同评测：

```bash
bash run_all.sh
```

两种方式二选一即可。`run_all.sh` 使用固定 `runs/sft`、`runs/grpo`、`runs/sft_grpo` 和评测名称；已经跑过时不要原地重复覆盖。`run_logged.sh` 拒绝覆盖同名日志，但这不代表每个底层 Python 入口都具备防覆盖保护。默认不自动恢复中断训练；独立重试应使用新输出目录。

## 6. 独立测试、查看进度与产物

```bash
source runtime.sh
python status.py

# baseline 使用相同测试入口；name 对应 results 下新目录
python evaluate_vllm.py --name base_new --split test
python audit_adapter.py runs/my_sft
adapter=$(python audit_adapter.py runs/my_sft --path-only)
python evaluate_vllm.py --name sft_new --adapter "$adapter" --split test

# 组合阶段交接的验收
python audit_adapter.py runs/my_sft_grpo/grpo_stage \
  --reference runs/my_sft_grpo/sft_stage

# 仅适用于 run_all 的标准目录/名称且训练评测都已完成
bash finalize_results.sh
```

`status.py` 的 `started_no_exit_record` 只说明有启动记录，不代表进程还活着。需要结合服务器进程和 `nvidia-smi` 判断，禁止看到无退出码就直接重复启动训练。

运行产物保留在服务器：

- `runs/`：LoRA、优化器状态、实际 Hydra 配置、RL rollout；
- `results/`：逐条预测、真实标签、样本 SHA、指标和评测配置；
- `reports/`：预检、环境/代码指纹、日志、退出码、每 2 秒物理显存采样和总结果。

这些目录默认不入 Git。推理耗时与整个阶段耗时不同，显存采样可能漏掉瞬时尖峰。仓库中的历史结果位于 `docs/results/2026-09-05/`，新实验写 `reports/`，两者不要混淆。

## 7. 后续扩大规模主要改哪些参数

完整说明见 [MIGRATION.md](MIGRATION.md)。`experiment_config.py` 定义默认值，JSON 只覆盖必要字段，未知字段会报错。

| 配置 | 定位 | 本次是否完成 |
|---|---|---|
| `workflow_3060.json` | 每阶段 4 步，测试每任务 4 条 | 是 |
| `dataset_3060.json` | SFT 3 epoch、GRPO 35 步、全部 40 条测试 | 否，仅提供配置 |
| `scale_single_node.example.json` | 单节点多卡、更大 dense Qwen3-VL 的参数模板 | 否，未实机验收 |

迁移顺序：环境和源码版本固定 → 配置新模型/数据路径 → 同配置小步验收 → 调整 batch、LoRA、分辨率、token、步数 → 正式多种子和完整测试。提高图像分辨率时必须同时检查 SFT、GRPO 和测试的 token 上限。

同 schema、同 dense 模型架构、单节点规模扩展可以复用入口，但**不是任何变化都只需改参数**：新增任务/schema 要改数据契约；MoE 要重新适配；多节点还需要 torchrun rendezvous、Ray 集群和调度器；自动断点恢复未实现。正式研究须重新做病例隔离，不能沿用本次图像 SHA 隔离来宣称病例泛化。

## 8. 可复核性与发布范围

[发布来源指纹](docs/results/2026-09-05/publication_provenance.json) 记录已完成远端脚本包的 SHA256 和逐文件指纹。三组核心训练代码、共同评测代码、配置和兼容插件在发布时保持原样；新增 README/数据契约/测试，文档链接和报告生成模板做了可迁移化处理。聚合结果只替换机器特定根路径，数值没有更改。

不将自动检查称为独立审计；不将小样本可执行性称为算法优越性；没有把未完成的全量/多卡实验写成结果。使用、分享真实 GraSP 数据或模型时，应分别核对数据、模型与上游软件授权。本仓库尚未由所有者指定独立开源许可证，参见 [NOTICE.md](NOTICE.md)。

上游参考：[VAGEN-Lite 固定提交](https://github.com/mll-lab-nu/VAGEN/tree/04bf4bd13bd93688d5cd66331745190486fd14d1)、[其 VERL 子模块来源](https://github.com/mll-lab-nu/VAGEN/blob/04bf4bd13bd93688d5cd66331745190486fd14d1/.gitmodules)、[vLLM 0.11.0 LoRA](https://docs.vllm.ai/en/v0.11.0/features/lora.html)。
