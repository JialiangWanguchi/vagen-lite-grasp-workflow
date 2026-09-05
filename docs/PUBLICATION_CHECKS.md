# 发布验收（2026-09-05）

以下在发布工作副本中执行，不消耗远端训练 GPU，不等同于重新训练：

| 检查 | 结果 |
|---|---|
| `python -m unittest -v test_experiment_config.py test_task_contract.py` | 16 项全部通过（原配置测试 8 项 + 新增纯文本任务契约测试 8 项） |
| `python task_contract.py` | TASK_CONTRACT_CHECKS_PASS |
| `python -m compileall -q .` | 所有 Python 文件语法编译通过 |
| 对全部 `.sh` 执行 `bash -n` | 全部通过 |
| 核心训练、评测、数据适配、配置与 compat 文件 SHA256 | 与已完成远端导出包一致 |
| 聚合结果的全部数值/布尔字段 | 与原始 `final_results.json` 完全相同 |
| 阶段退出码 | 与原始私有退出码文件核对一致，包含外层失败及汇总修复 |
| 发布目录范围与敏感模式扫描 | 未含模型、真实数据/图像、预测、原始日志、聊天、服务器登录信息或凭据 |

与已完成脚本包相比，仅 `ENVIRONMENT.md`、`WORKFLOW_REPORT.md` 和报告生成器中的示例 `cd` 路径改变。README、数据契约、授权说明、发布测试与 `.gitignore` 为新增；训练逻辑未改。

本次没有重新在全新机器安装全部依赖，没有重新进行 GPU 训练，没有完成独立审计，也没有测试全量训练、多卡或多节点。此页的 CPU 检查不替代真实数据预检和新硬件短跑。
