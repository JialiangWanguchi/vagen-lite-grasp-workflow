# 同一套脚本的小规模验证与迁移

本工程当前的目标是用真实 GraSP A2/P1 数据验证训练、权重交接、推理与测试流程，不是用这 200 条数据证明算法优劣。三组都是 VERL/FSDP 梯度训练，vLLM 生成和测试；GRPO 通过固定提交的 VAGEN-Lite 单轮环境执行。

## 独立入口

在远端工程目录执行，模型与图片均留在远端。

```bash
# 默认读取 profiles/workflow_3060.json
bash run_sft.sh --output runs/my_sft
bash run_grpo.sh --output runs/my_grpo
bash run_sft_grpo.sh --output runs/my_sft_grpo

# 完整脚本级验收：基座评测 + 三组训练、适配器校验和共同评测
bash run_all.sh

# 断开后查询进度；没有退出码仅表示曾启动，不保证进程仍存活
source runtime.sh
python status.py

# 训练和评测已经完成，只重建报告（不重新训练）
bash finalize_results.sh
```

组合组不是复用第一组的结果：它独立从基座做 SFT，再将实际导出的 LoRA 传给 GRPO。审计要求组合 GRPO 权重相对其 SFT 阶段产生非零变化。独立输出目录应使用新名字；run_all 的固定输出用于本次验收，请勿重复覆盖。

## 配置方式

`experiment_config.py` 定义完整默认值，JSON 文件只覆盖要改的字段。未知字段会报错，避免拼写错误被忽略。模型、图片、数据路径可使用绝对路径，或相对于 `GRASP_ROOT` 的路径。

```bash
export GRASP_CONFIG="$PWD/profiles/dataset_3060.json"
bash run_sft.sh --output runs/sft_dataset
bash run_grpo.sh --output runs/grpo_dataset
bash run_sft_grpo.sh --output runs/combined_dataset
```

| 配置 | 用途 | SFT | GRPO | 评测 |
|---|---|---|---|---|
| workflow_3060.json | 本次脚本级验收 | 4 步 | 4 步 | 每任务固定前 4 条测试样本，共 8 条 |
| dataset_3060.json | 当前 200 条数据更长的试跑 | 3 epoch | 35 步 | 全部 40 条测试样本 |
| scale_single_node.example.json | 大 GPU/大模型参数模板，未实机验证 | 有效批量 32 | 有效批量 32、组大小 4 | 全量 |

三个训练组使用同一份 train/val/test 文件。短跑只是限制优化器步数，并不从测试集选择训练样本；8 条评测样本按各任务已有顺序固定选取，不按结果挑选。所有组使用相同评测样本、图像处理与贪心解码。

每次训练保存 `resolved_config.yaml` 和 `experiment_profile.json`；测试保存逐条预测、指标和配置。`run_logged.sh` 保存退出码、时间、日志和每 2 秒采样的物理显存占用，拒绝覆盖已有同名日志。

## 换服务器时

1. 复制工程代码、`profiles/`、`compat/`，并在新远端安装固定的依赖与源码；原始数据和权重由服务器间传输或在新服务器下载，无需经本机中转。
2. 设置新环境路径。`GRASP_ROOT` 默认就是脚本所在目录，不再固定为本次 `/workspace/...`；也可设置 `GRASP_VENV`、`VAGEN_DIR`、`VERL_DIR`。
3. 复制大规模模板，改 `model`、`data.prepared_dir`、`data.image_root`、图像分辨率、token 上限、LoRA、批量、epoch/steps 和 GPU 参数。
4. 先运行 `source runtime.sh && python preflight.py`，再各运行短步数版本；通过真实参数更新和权重重载后才延长训练。

```bash
export GRASP_ROOT=/new/server/grasp_project
export GRASP_VENV=/new/server/grasp_project/venv
export GRASP_CONFIG="$GRASP_ROOT/profiles/my_scale.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$GRASP_ROOT"
source runtime.sh
python preflight.py
bash run_sft.sh --output runs/scale_sft
```

同一单节点的 SFT 进程数由 `hardware.gpus_per_node` 配置；GRPO 读取相同 GPU 数和 `tensor_parallel`。当前实测仅单张 RTX 3060，4 卡模板只是配置示例，不代表多卡验收通过。多节点还需要调度器、torchrun rendezvous 和 Ray 集群管理；不能承诺只改一个 GPU 数就完成多节点部署。

## 需要保留的技术边界

- 已适配的是 dense `qwen3_vl` 架构。更大的同架构 Qwen3-VL 可复用入口，但仍必须预检。文本 Qwen3 不接收图片；Qwen3-VL MoE/其他架构不能仅靠改名字保证兼容。
- 当前视觉 SFT 每卡微批量固定为 1，有效批量通过原生梯度累积扩大。适配器显式要求每个 microbatch 一个样本，不再依赖 224 分辨率样本恰好超过某个 token 长度。
- 图像分辨率由 `vision` 配置统一控制，所有帧保留，不静默截断图片。升高分辨率需要同时提高 SFT/GRPO/评测 token 上限；预检会检查数据实际长度。
- LoRA rank/alpha 已参数化，去掉只适用于 2B 的固定参数总数断言；三组最终仍需核对实际适配器容量、有限值、非零更新及权重交接。
- 严格奖励仍是原始 JSON 合法且答案精确匹配才得 1，没有为凑成功而改成格式奖励。组内奖励全相同的批次梯度可能为零，应在报告中区分“步骤执行”与“有效更新”。
- 数据适配器仍使用现有 GraSP JSONL schema 与 A2/P1 任务。扩大同 schema 数据只需换路径；新增任务/改 schema 必须扩展 `task_contract.py`，不能靠参数虚构兼容。
- `prepare_remote.py` 是本次 200 条数据的特定分组划分脚本，不是通用大规模划分器。迁移时提供经独立审计的 train/val/test JSONL；特别要在正式研究中做到病例隔离。当前图像 SHA 隔离不等于病例隔离。
- 已固定环境：PyTorch 2.8.0、Transformers 4.57.1、vLLM 0.11.0、VERL 提交 `3fe0a29975e1b02ae2bd1dec249f7807dd7966f5`、VAGEN-Lite 提交 `04bf4bd13bd93688d5cd66331745190486fd14d1`。版本升级需重测兼容补丁，不能直接沿用测试结论。
- 本版本默认不自动恢复中断训练；新进程请使用新输出目录。自动断点恢复不是本次已经验收的能力。

官方参考：[VAGEN-Lite 分支](https://github.com/mll-lab-nu/VAGEN/tree/vagen-lite)、[vLLM 0.11.0 LoRA 文档](https://docs.vllm.ai/en/v0.11.0/features/lora.html)。实际验收以固定版本的本地源码和远端日志为准。
