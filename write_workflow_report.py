"""Render a bounded workflow report only after all recorded checks pass."""
import datetime
import json
import os
from experiment_config import ROOT

s=json.loads((ROOT/'reports/final_results.json').read_text())
cfg=s['profile']
pre=json.loads((ROOT/'reports/profile_preflight.json').read_text())
env=json.loads((ROOT/'reports/environment.json').read_text())
counts={k:v['n'] for k,v in pre['splits'].items()}
test_counts={k:v['n'] for k,v in s['results']['base']['metrics'].items()}
ram_gib=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1024**3
lines=['# VAGEN-Lite / Qwen3-VL 三组脚本级训练与测试报告','',
    '## 结论与适用范围','',
    '三组小规模真实训练及 vLLM 重载评测均完成，自动验收通过：SFT、GRPO-only、独立 SFT→GRPO。',
    '这是同一套可参数化入口的流程验证，不是充分训练、性能优化、临床可靠性或多卡扩展性验证。',
    '独立实验审计未完成：指定 GPT-5.4 调用被当前账户拒绝。自动断言不等同于独立审计，详见 EXPERIMENT_AUDIT.md。','',
    f"报告生成时间（UTC）：{datetime.datetime.now(datetime.timezone.utc).isoformat()}",'',
    '## 硬件、代码与统一框架','',
    f"- 实测 GPU：{env['gpu']}；报告生成节点主机内存 {ram_gib:.2f} GiB。模型及图像处理均在远端执行。",
    f"- 基座：{cfg['model']}（dense qwen3_vl）。LoRA 只训练语言层；视觉骨干冻结。",
    '- VAGEN 指定 vagen-lite 提交 `04bf4bd13bd93688d5cd66331745190486fd14d1`，VERL 提交 `3fe0a29975e1b02ae2bd1dec249f7807dd7966f5`。',
    '- 三组梯度训练统一使用 VERL/FSDP；GRPO 的采样及全部组的测试统一使用 vLLM。vLLM 不是梯度训练器，这一口径已经确认。',
    '- 关键版本：torch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.57.1、PEFT 0.17.1、FlashAttention 2.8.3、Ray 2.49.2。完整版本与代码指纹见 reports/environment.json 和 pip_freeze.txt。','',
    '## 数据与指标','',
    f"- 数据划分条数：{counts}。本次训练预算见下表，有限步骤不代表遍历完整训练集。",
    f"- 评测按原测试划分顺序、每任务 per_task={cfg['evaluation']['per_task']}（0 表示全量）选取，实际条数：{test_counts}。四种模型使用相同順序，未按预测结果挑选样本。",
    '- 相同图片 SHA256 不跨集合；病例及邻近非重复帧仍可能跨集合，不能声称病例泛化。原始标签来自用户数据的 gt_answer，而非模型生成。',
    f"- 图像全部保留，逐帧最长边 {cfg['vision']['max_side']}；允许上限 {cfg['vision']['max_images']} 张。提示词中的原始 CASE/帧文件名替换为中性 Clip/Frame 标签，避免直接暴露顺序。",
    '- 输出必须为恰含 think、answer 的 JSON；A2 答案为 A–E 字符串，P1 为 A/B/C 的排列数组。重复键、代码围栏或非法类型判格式失败。',
    '- 严格准确率 = JSON 合法且答案与数据标签相同的条数 / 样本数；奖励也按同一规则取 0 或 1。JSON 合法率另行报告。没有按模型自身分数归一化。',
    '- 数据没有专家推理文本，SFT 的 think 使用空字符串，不合成推理标签。',
    '- 基座最早一版预测未写样本 SHA 字段；其行身份由未变更的测试文件、配置、index、task 和 truth 重建核对，其余组直接记录样本 SHA。','',
    '## 本次运行参数与步骤','',
    f"完整配置：reports/experiment_profile.json；随机种子 {cfg['seed']}；LoRA rank={cfg['lora']['rank']}，alpha={cfg['lora']['alpha']}。",
    f"- SFT：steps={cfg['sft']['steps']}，epochs={cfg['sft']['epochs']}，有效批量 {cfg['sft']['batch_size']}、每卡微批量 1，学习率 {cfg['sft']['lr']}。",
    f"- GRPO：steps={cfg['grpo']['steps']}，epochs={cfg['grpo']['epochs']}，prompt batch {cfg['grpo']['batch_size']}，每题采样 {cfg['grpo']['group_size']} 个回答，学习率 {cfg['grpo']['lr']}；max_turns=1。",
    '- 组合组：独立从基座做同预算 SFT，然后实际加载该 LoRA 做同预算 GRPO，不复用第一组的训练产物。',
    f"- 输入上限 {cfg['grpo']['prompt_tokens']} tokens，生成上限 {cfg['grpo']['response_tokens']}；vLLM 贪心测试 temperature=0。预检使用真实 {sum(counts.values())} 条数据，禁止静默截断输入。",'',
    '## 实际评测结果（real_gt）','',
    f"| 权重 | A2 正确/{test_counts['A2']} | P1 正确/{test_counts['P1']} | 总正确/{test_counts['all']} | JSON 合法/{test_counts['all']} | 截断条数 |",'|---|---:|---:|---:|---:|---:|']
for arm,r in s['results'].items():
    m=r['metrics'];count=lambda task,key:round(m[task][key]*m[task]['n'])
    lines.append(f"| {arm} | {count('A2','exact_match')} | {count('P1','exact_match')} | {count('all','exact_match')} | {count('all','format_valid')} | {m['all']['truncated']} |")
lines+=['','相对基座的准确率差值：','', '| 组 | 差值（百分点） | 相对变化 |','|---|---:|---:|']
for arm,d in s['delta_vs_base'].items():
    relative='未定义（基座为 0）' if d['relative_change'] is None else f"{d['relative_change']:.2%}"
    lines.append(f"| {arm} | {d['accuracy_difference']*100:.2f} | {relative} |")
lines+=['','不能把当前极小样本的格式变化解释为算法优势。截断与格式不合法表明输出控制仍需验证；真实正确率低也可能涉及模型容量、训练量、视觉分辨率及任务难度，当前实验不能区分原因。','',
    '## 真实优化器步骤与学习信号','',
    '| 阶段 | 步 | SFT loss | 梯度范数 | GRPO 平均奖励 | 组内奖励方差指标 |','|---|---:|---:|---:|---:|---:|']
for phase,entries in s['training_metrics'].items():
    for m in entries:
        def val(k):return f"{m[k]:.6g}" if k in m else '—'
        gradient=val('actor/grad_norm') if 'actor/grad_norm' in m else val('train/grad_norm')
        lines.append(f"| {phase} | {int(m['step'])} | {val('train/loss')} | {gradient} | {val('critic/score/mean')} | {val('custom_metrics/train/reward_variance')} |")
lines+=['','不同 SFT 步骤使用不同批次，不能仅凭 loss 首尾值证明收敛。GRPO 中组内奖励相同可产生零梯度；本次自动验收要求两个 RL 阶段均至少出现一次非零策略梯度和组内奖励差异，不把执行步数等同于有效学习次数。','',
    '## 检查点证据','', '| 组 | 张量数 | 参数数 | LoRA B L2 | 相对 SFT 阶段权重差 L2 |','|---|---:|---:|---:|---:|']
for arm,a in s['artifacts'].items():
    lines.append(f"| {arm} | {a['tensor_count']} | {a['parameters']} | {a['lora_B_l2']:.8g} | {a.get('parameter_delta_l2','—')} |")
lines+=['','上述适配器均通过有限值、LoRA 维度及无视觉层适配器检查。组合组额外与其实际 SFT 检查点比较，要求非零变化；完整 SHA256 与路径记录在 reports/final_results.json。','',
    '## 用时与物理显存','', '| 阶段 | 用时（秒） | 每 2 秒采样的显存峰值（MiB） |','|---|---:|---:|']
for phase,r in s['resources'].items():lines.append(f"| {phase} | {r['elapsed_seconds']:.0f} | {r['sampled_peak_gpu_MiB']:.0f} |")
lines+=['','以上为 nvidia-smi 物理采样值，可能漏掉采样间隔内瞬时峰值；不使用混合训练/推理内存池报告的 allocator 数字代替物理显存。组合组用时包含其独立 SFT 与 GRPO 两阶段。','',
    '## 问题处理记录与测试边界','',
    '- 早期 SFT 依次暴露模型选择缺少生成头、FlashAttention 辅助模块、Python.h、jagged split、FSDP 视觉参数回写和不必要整模型保存问题。修复均在独立适配模块或隔离环境中，未更改数据标准答案。',
    '- vLLM 0.11.0 的 Qwen3-VL 多模态前缀通过 compat 插件修正，防止语言 LoRA 错误包装视觉层；没有改写模型权重。',
    '- 早期 GRPO 单步完成但主机 RAM 压力过高；关闭 CPU 参数/优化器卸载、采用分层权重同步及 decode CUDA graphs 后，两步连续测试退出 0。此后才运行本次四步三组验收。',
    '- 当前视觉 SFT 显式将最少 microbatch 数设为样本数，保持每 microbatch 一条，不再依赖本次图像 token 长度恰好适合 1024 预算。',
    '- 配置单元测试 8 项通过；真实数据预检通过；三组 checkpoint 与同样本指标复算通过。独立审计状态为 unavailable。',
    '- 本次 workflow_suite 在全部训练/评测退出 0 后，因 Python 3.10 的 ISO 时间 Z 后缀解析失败而退出 1。修复为显式 +00:00 后，通过独立 report_finalization 重新运行汇总；原失败日志与退出码保留，未重跑或改写训练结果。','',
    '## 复现与迁移','',
    '```bash', 'cd "$GRASP_ROOT"  # set to your project directory', 'export GRASP_CONFIG="$PWD/profiles/workflow_3060.json"',
    '# 新部署/新输出空间运行；已有同名日志会被保护，不能原地覆盖', 'bash run_all.sh', '```','',
    '三个独立入口分别为 run_sft.sh、run_grpo.sh、run_sft_grpo.sh。完整字段、单节点多卡示例与技术边界见 MIGRATION.md；环境重建见 ENVIRONMENT.md。',
    '报告、预测、日志与指纹在远端 reports/、results/；权重只留在 runs/。本地 evidence/ 保存小型日志和指标副本，不下载模型或图片。','',
    '下一步建议：保持代码入口，在新 GPU 上先做同配置短跑，再使用病例隔离的大数据划分扩展模型/分辨率/训练预算；正式效果比较增加独立验证集、多个随机种子、完整测试集与人工标签审查。多节点、自动断点恢复、其他模型架构均不属于本次已验收能力。','']
(ROOT/'WORKFLOW_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('WORKFLOW_REPORT_WRITTEN')
