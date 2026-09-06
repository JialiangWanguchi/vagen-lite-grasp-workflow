# GraSP 数据划分方案 v2

## 当前 200 条数据的审计结论

对 A2、P1 各 100 条源数据做病例共现图审计：

| 项目 | 结果 |
|---|---:|
| 总样本 | 200 |
| A2 / P1 | 100 / 100 |
| 唯一病例 | 13 |
| A2 涉及病例 | 13 |
| P1 涉及病例 | 8 |
| 每条 A2 病例数 | 4 |
| 每条 P1 病例数 | 1 |
| 病例共现连通分量 | 1（含全部 13 个病例） |

所以，现成 A2 行无法被分到三个非空且病例互斥的集合：只要两个病例在同一 A2 问题出现，它们就必须进入同一 split；传递闭包最终把 13 个病例全部绑在一起。以前按样本/图片 SHA 得到的 140/20/40 只适合脚本冒烟测试，不能支持“未知病例泛化”的结论。

## 推荐的 case-first 流程

1. 在生成问题前提取并冻结病例清单；源数据当前有 13 个病例。
2. 固定 seed `20260906`，划为 train/val/test = 5/4/4 个病例。每份至少 4 个病例，因为一条 A2 需要 4 个不同病例。
3. 确保每份至少有一个能生成 P1 的病例。
4. 仅在各自病例池内部重新生成 A2；P1 也只能使用所属池内的单个病例。
5. 每个任务分别以 70/10/20 条为目标，总计 train/val/test = 140/20/40。病例数比例不等于样本数比例，训练病例池可重复采样生成更多组合。
6. 去重时同时检查病例 ID、精确图片 SHA-256、帧路径和感知近重复；任何跨 split 命中均为失败。
7. 先冻结 train/val/test manifest 及 SHA，再训练。长度上限和 fallback 阈值只用 train/val 调整；test 保持封存。

13 个病例只能提供很粗的病例覆盖，正式效果研究应扩大病例数。当前 5/4/4 适合验证新流程是否无泄漏，不适合窄置信区间的临床泛化结论。

## 工具

```bash
# 1) 审计现有行的病例共现图
python case_split.py audit \
  --input A2_GraSP_final.jsonl --input P1_GraSP_final.jsonl \
  --output case_audit.json

# 2) 在生成问题前建立病例 manifest（输出不应提交公开仓库）
python case_split.py plan \
  --input A2_GraSP_final.jsonl --input P1_GraSP_final.jsonl \
  --seed 20260906 --output private_case_manifest.json

# 3) 用 manifest 检查并物化“重新生成后”的样本
python case_split.py materialize \
  --input regenerated_A2.jsonl --input regenerated_P1.jsonl \
  --manifest private_case_manifest.json --output-dir prepared_v2
```

第三步对当前旧 A2 数据应当失败，这是正确行为；它会列出跨病例分区的样本示例，而不是静默泄漏。

## 验收门槛

- train/val/test 的病例集合两两交集为零；
- 图片 SHA、规范化帧路径、感知近重复跨集合为零；
- A2、P1 在三个 split 都非空；
- 每条 A2 的四个病例属于同一 manifest split；
- manifest、生成器版本、seed、源数据指纹写入运行产物；
- 校准阶段不读 test，最终比较才读 test。

