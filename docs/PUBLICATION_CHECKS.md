# 发布验收（2026-09-05）

以下在发布工作副本中执行，不消耗远端训练 GPU，不等同于重新训练：

| 检查 | 结果 |
|---|---|
| `python -m unittest -v test_experiment_config.py test_task_contract.py test_evaluation_budget.py` | 21 项全部通过（配置 8 项 + 任务契约 8 项 + 评测预算 5 项） |
| `python task_contract.py` | TASK_CONTRACT_CHECKS_PASS |
| `python -m compileall -q .` | 所有 Python 文件语法编译通过 |
| 对全部 `.sh` 执行 `bash -n` | 全部通过 |
| 核心训练、评测、数据适配、配置与 compat 文件 SHA256 | 与已完成远端导出包一致 |
| 聚合结果的全部数值/布尔字段 | 与原始 `final_results.json` 完全相同 |
| 阶段退出码 | 与原始私有退出码文件核对一致，包含外层失败及汇总修复 |
| 发布目录范围与敏感模式扫描 | 未含模型、真实数据/图像、预测、原始日志、聊天、服务器登录信息或凭据 |

与已完成脚本包相比，仅 `ENVIRONMENT.md`、`WORKFLOW_REPORT.md` 和报告生成器中的示例 `cd` 路径改变。README、数据契约、授权说明、发布测试与 `.gitignore` 为新增；训练逻辑未改。

本次没有重新在全新机器安装全部依赖，没有重新进行 GPU 训练，没有完成独立审计，也没有测试全量训练、多卡或多节点。此页的 CPU 检查不替代真实数据预检和新硬件短跑。

## 2048-token 补充评测

- 远端真实数据预算预检、base/SFT/GRPO/SFT→GRPO 评测、外层 suite 和配对比较共 7 个退出码，全部为 0。
- 复算 32 条新输出的格式与真实标签指标；核对同一测试行、输入 token 数与样本 SHA。
- 三个 LoRA 的权重 SHA256 与原始实验完全一致；训练未被调用。补充实验后默认评测预算同步更新为 2048/4096，历史配置留在证据目录。
- 完成后 GPU 为 2 MiB / 0% 利用率，无遗留 screen 会话。
- GitHub 只发布聚合对比、配置、样本 SHA 和退出码，不发布逐条预测、图片、标签或原始日志。

## 2026-09-06 长度与判分 v2

- 2048 现仅作为诊断 profile；四组各 20 条验证输出完成后，默认评测上限按冻结规则改为 1024。
- 长度硬判负、严格/fallback 两阶段判分、病例 manifest 工具已加入 CPU 回归测试。
- VAGEN-Lite agent loop 的长度元数据桥接已在远端环境导入验证；真实 `GraSPEnv.step` 的长度 0 分与 fallback 0.5 分检查通过。
- 现有 200 行的病例共现审计得到 13 个病例、单一连通分量；旧行不能形成病例互斥三分，工具拒绝了 99 个跨分区样本。
