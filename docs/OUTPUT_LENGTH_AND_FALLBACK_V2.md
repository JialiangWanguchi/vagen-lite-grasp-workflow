# 输出长度与 fallback 判分方案 v2

## 结论

长度上限不再凭测试集表现手工选择。四个模型组先在固定验证集 20 条样本上，以 2048 tokens 作为诊断上限生成 80 个输出；排除所有 `finish_reason=length` 的输出后，统计均值、中位数、P90、P95 和最大值。2026-09-06 的 80 条远端校准已完成：21 条触顶被排除，59 条正常输出均值 249.54、中位数 224、P90 468、P95 630、最大值 820，因此最终评测上限为 **1024**。完整结果见 `docs/LENGTH_CALIBRATION_REPORT.md`。

```text
smallest k * 512 strictly greater than max(normal validation output_tokens)
```

例如正常输出最大值为 1135，则选择 1536。测试集不参与该参数选择，只在参数冻结后使用一次。`calibrate_output_length.py` 会拒绝 `--split test`，防止无意中用测试集调参。

此前 8 条测试样本的 2048-token 复跑只能作为回顾性诊断：26 个正常结束输出均值 267.85、中位数 260.5、最大值 505；它不能用于正式选择上限。正式数值应以新的验证集校准文件为准。

## 超长硬判负

判分顺序固定为：

1. 先读取生成引擎的可信长度元数据；
2. 若 vLLM `finish_reason == "length"`，直接 reward=0；
3. VAGEN rollout 若本轮 `len(response_ids) >= response_limit`，也直接 reward=0；
4. 只有正常结束的输出才进入严格 JSON 和 fallback 判断。

文本中是否碰巧出现了正确字母，不会覆盖长度硬判负。评测记录保留 `content_exact_before_length` 供审计，但 `exact_match`、`accepted_match` 和 reward 均为负。

## 两阶段判分

严格层保持原协议：输出必须是且只能是 `think`、`answer` 两个字段的 JSON 对象；A2 使用大写 A–E，P1 使用 A/B/C 的三元素数组。严格正确 reward=1.0。

fallback 只处理高置信度、语义唯一的格式变体，正确时 reward=0.5：

- A2：`a`、`Option a`、`Clip A`、`answer is a`、JSON 中的小写答案；
- A2 的 E：`None of the above` 或“all four clips ... same surgical phase”等预先列举的等价短语；
- P1：`B,C,A`、`B > C > A`、`Clip B then Clip C then Clip A`、小写 JSON 数组；
- 完整且闭合的 JSON code fence。
- JSON 对象整体闭合但 `think` 中存在未转义字符时，可从唯一的末端 `answer` 字段救回；重复 `answer` 不救回。

以下情况不自动救回：

- 未闭合 JSON/code fence，或没有唯一可验证 `answer` 字段的解析失败对象；
- 从长篇 reasoning 中搜索任意 A/B/C 字母；
- 同时给出多个答案或多个顺序；
- P1 缺项、重复项或非法项；
- 超过长度上限的任何输出。

歧义输出标记 `review_required=true`，线上训练 reward=0，不调用外部 LLM judge。这样 fallback 是确定性、可复现的，且不会因 judge 模型漂移改变奖励。

## 报告指标

每次评测同时报告：

- `strict_exact_match`：严格格式且答案正确；
- `accepted_match`：严格正确或高置信 fallback 正确；
- `format_valid`：原始严格 JSON 合法率；
- `fallback_correct`：被 fallback 正确救回的数量；
- `hard_negative_length`：触及上限、被硬判负的数量；
- `review_required`：需人工复核但未自动给分的数量。

不能只展示 `accepted_match`；严格指标用于观察模型是否真正学会输出协议。

## 使用

```bash
# 在远端、已有三组 adapter 的前提下，运行四组验证集长度校准
bash run_length_calibration.sh lengthcal_v2

# 输出
cat results/lengthcal_v2_recommendation.json
```

校准 profile 的诊断上限为 2048。若仍有大量正常样本触顶，应先提高诊断上限并重跑验证集校准；不能把触顶值当作自然输出最大值。
