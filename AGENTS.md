# 项目协作规则

- 未特别指定时，文档与结果说明使用中文。
- 每次完成修改后，核对当前分支、远程跟踪分支及待推送提交，只提交本次任务涉及的文件，直接推送到对应远程。交付时报告提交和推送结果。遇到冲突或权限阻碍时说明情况，保留其他在途工作。
- 训练或大批量测试只提供 Windows CMD 命令，不由代理直接运行；允许执行单元测试和单样本 smoke test。
- PyTorch 训练、批量评测和模型诊断默认使用 CUDA GPU。给出 CMD 命令前先确认解释器支持 CUDA，脚本支持设备参数时显式指定 `--device cuda`。本机已验证 `D:\anaconda\envs\lightglue_env\python.exe` 可在 RTX 4060 Ti 上运行 DCR-Synth，项目 `.venv` 当前使用 CPU 版 PyTorch。GPU 不可用时说明原因，未经用户同意不回退到 CPU。
- 每次新增模块后的完整训练默认从头初始化整套模型，沿用架构规定的固定算子初始化。除非用户明确要求微调或指定初始化来源，否则不加载 R3、E2 等已训练的中间模型权重。同一次训练的断点续训不属于从中间模型初始化。
- 默认复用已有训练 checkpoint、基线和适用的评价结果，不默认安排额外重复训练。设计实验、脚本和推荐命令前先检查现有结果；脚本默认只运行新增或缺失的实验，不将重训基线包含在默认流程中。不能仅以“同一代码环境下更严格”为由默认重训；若发现影响可比性的具体差异，应先说明差异、复用限制及额外成本，再由用户明确决定是否重训。
- 后续光学–SAR 融合的验证和测试统一使用 `reference_v2_candidate_with_paper_fmi`。FMI 采用 `PaperEvaluator.FMI` 的全图 DCT 互信息之和，其余 13 项采用 `ReferenceEvaluator`。新评测使用独立比较组，旧协议用于复现历史实验。
- 多方法比较对每项指标使用各方法共同具有有限值的场景，保留逐图原值，并报告共同样本数。MS-SSIM 出现非有限值时按同一规则处理。
- 每次新增训练、验证、确认或正式测试结果后，必须更新 `EXPERIMENT_REGISTRY.json`。
- 每次模型、损失、推理路径、数据切分或评价协议变化后，测试必须写入新的唯一输出目录，禁止覆盖旧目录。
- 更新注册表后必须执行：

```cmd
.venv\Scripts\python.exe maintain_experiment_index.py --check --render
```

- 只有 `comparable_group` 完全相同的实验才能放入同一张定量对比表。
- 数据集、split、offset、样本数、checkpoint、metric profile 或输出策略任一不同，必须新建比较组，不得直接混用数值。
- 每个正式评价条目必须保存：checkpoint 路径与 SHA-256、split manifest 路径与 SHA-256、原始指标文件、运行元数据和结果目录。
- 每次正式 test1000 必须保存全部1000张灰度融合图、RGB融合图和 Optical/SAR/Fused 三联预览，不得只保存少量样例或使用 metrics-only。
- 每次正式 test1000 必须输出完整14项融合指标的逐图结果、均值和标准差，并更新五方法对比表：B0、B0+SPD、B0+Detail-only-SPD、C0-minimax、C0-minimax+SPD。
- `EXPERIMENT_REGISTRY.json` 是机器可校验事实源；`EXPERIMENT_INDEX.md` 与 `TEST_OUTPUT_LOG.md` 由脚本生成，不手工修改生成表格。
