# 项目协作规则

- 未特别指定时，文档与结果说明使用中文。
- 训练或大批量测试只提供 Windows CMD 命令，不由代理直接运行；允许执行单元测试和单样本 smoke test。
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
