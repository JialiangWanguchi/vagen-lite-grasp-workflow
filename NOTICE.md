# 来源与授权边界

本仓库是按所有者要求公开的实验工作流仓库，不是 VAGEN、VERL、vLLM 或 Qwen 官方发行版。项目所有者尚未指定本仓库新增代码的独立开源许可证；仓库可公开访问不等于自动授予任意再分发许可。

依赖分别从上游安装，不在这里重新分发完整上游源码：

- VAGEN：`mll-lab-nu/VAGEN`，`vagen-lite`，提交 `04bf4bd13bd93688d5cd66331745190486fd14d1`。
- VERL：VAGEN 固定的 `JamesKrW/verl` 子模块，提交 `3fe0a29975e1b02ae2bd1dec249f7807dd7966f5`。
- vLLM、Transformers、PyTorch、PEFT、FlashAttention 等保留各自上游授权要求。
- Qwen 模型及 GraSP 数据不随仓库分发，应分别核对模型和数据授权。

兼容适配模块依赖上述固定版本的接口。正式公开发布或再分发前，请由所有者审查上游许可证与衍生代码义务，并决定本仓库许可证。
