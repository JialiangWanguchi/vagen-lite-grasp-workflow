# 数据契约与拆分责任

本仓库不分发用户数据。训练和测试接收 UTF-8 JSONL，每行一个样本；三个 split 文件都需要包含 A2 和 P1。训练入口不自动读取原始压缩包，也不自动生成正式研究所需的病例隔离划分。

## 必要字段

| 字段 | 类型 | 约束 |
|---|---|---|
| `task_id` | 字符串 | `A2` 或 `P1` |
| `question` | 字符串 | 原始问题；按 clips/frames 顺序包含每个 `image_path` 文本 |
| `gt_answer` | 字符串或数组 | A2：`A`–`E`；P1：A/B/C 排列数组或可解析成该数组的 JSON 字符串 |
| `clips` | 数组 | 每个 clip 有 `clip_label` 和 `frames` |
| `clips[].frames[].image_path` | 字符串 | 对应实际图片；必须也出现在 question 中 |

其他字段可保留，但不要把标签线索写入提示词。`prompt_parts` 用中性 `Clip X, Frame N` 替换问题中的文件路径，并在同一位置插入图片。代码拒绝渲染后仍含 CASE 数字、`.jpg` 或 `data/GraSP` 的提示词。

图片路径的 `data/GraSP/` 前缀被去掉，再拼接 profile 的 `data.image_root`；因此 `data/GraSP/example/f1.jpg` 对应 `<image_root>/example/f1.jpg`。不要传入不可信绝对路径或 `..` 越界路径。

下面仅展示 schema，**不是实际训练数据，也不是已验证的完整任务样本**：

```json
{
  "task_id": "A2",
  "question": "Inspect the images and choose A-E.\ndata/GraSP/example/f1.jpg\nReturn only a JSON object with think and answer.",
  "gt_answer": "A",
  "clips": [{"clip_label": "A", "frames": [{"image_path": "data/GraSP/example/f1.jpg"}]}]
}
```

本次真实 A2 每题为 4 个 clip × 4 帧，共 16 图；P1 每题为 3 个 clip × 4 帧，共 12 图。通用加载器没有强制这两个固定张数，但会保留所有输入帧，并由预检限制总图片数和实际 token 长度。

## SFT、奖励与测试一致性

SFT 目标是 `{"think":"","answer":真实标签}`。数据不含专家推理，因此不伪造 rationale；例如 P1 可以是 `{"think":"","answer":["B","C","A"]}`。

生成结果必须是恰含 `think`、`answer` 两个键的 JSON 对象；`think` 必须是字符串。重复键、代码围栏、多余键、P1 字符串形式答案、重复或缺失的排列元素均判格式失败。只有合法且精确匹配真实标签时奖励为 1，否则为 0；合法率独立报告，不额外给予格式奖励。

所有组使用同一 `task_contract.py` 和 `grasp_common.py`。不修改真实答案来提高通过率，不用测试集奖励挑选提示词或模型。

## 本次拆分与未来扩大数据

随机种子 20260905；按相同图片 SHA 连通分量分组，划分为 140/20/40；每任务各 70/10/20。完全相同图片不跨集合，但同病例、邻近帧可能跨集合。本次只验证脚本可执行性，不支持病例泛化结论。

新数据应先按病例/视频来源等实体完成划分，再检查重复图像、近重复帧和标签质量。提供新 split 路径即可继续使用同 schema 训练入口；新增任务则要扩展标签校验、奖励、指标和预检中的任务枚举。

`preflight.py` 会处理三个 split 中的全部样本；更大数据上可能耗时较长，当前没有自动缓存或抽样预检功能。
