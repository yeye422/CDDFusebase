# FCSR 前置诊断：冻结 E2 Decoder 的频带混合与结构传递

日期：2026-11-18。状态：已实现冻结诊断脚本与合成单元测试，**尚无真实 QXS 运行结果**。不训练 FCSR，不改 E2 权重，不访问 test1000。

## 研究问题

E2 的实际重建为 concat(Base, Detail) → reduce_channel (1×1) → 4 个 Restormer TransformerBlock → output conv → Sigmoid。内部没有固定低/高频职责约束，但不能由此认定存在有害结构损失。

本诊断分为：

1. **频带传输**：相同 RMS 的 Base/Detail coarse/middle/fine 变化，在 reduce、transform、logits、image 四阶段产生多少高频响应？
2. **通道混合**：原有 1×1 卷积对 Base/Detail 的独立线性贡献是否在数值上抵消？这不是有效结构被删除的直接证据。
3. **源图候选边缘**：在原光学/SAR 单源重建中清晰、在融合图中减弱的候选是否存在？沿源特征方向的小幅插值能否恢复响应？这不能独立归因给 Decoder。

## 实验协议

- 使用归档模型 models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth、锁定 val64 manifest；
- 默认在 val64 中均匀抽取16张，可设为完整64张。仅做已训练 E2 冻结前向；
- 对融合 Base/Detail 各做非正交但严格可加的 2×/4× pooling 尺度分解；
- 逐 band 采用中心差分。6组探针的扰动 RMS 统一设为 0.015 × pooled RMS(Base,Detail)，记录四阶段响应 RMS、增益及一层正交 Haar 高频能量份额；
- 在原 reduce_channel 1×1 卷积上分别计算 Base/Detail 两路贡献，并报告余弦、能量比和输出饱和度；
- 用平滑 Sobel 提取源图显示域候选：一源内部梯度高15%、另一源低50%；SAR 有效像素基于 SAR>0。高梯度不是真实结构标签；
- 单源重建和融合图都使用完全相同的冻结 E2 Decoder。比较候选区域的有符号梯度传递；
- 额外对融合 Base/Detail 沿光学源特征方向分别进行4%小幅插值，仅测敏感性，不推定提升画质；
- 默认保存4张 Optical / SAR / E2 / Optical Reconstruction / SAR Reconstruction 预览。

## 与已有诊断的区别

原 diagnose_e2_information_path.py 已检查 Base/Detail 整体幅度变化及 Detail 三尺度响应。新版增加 Base/Detail **等 RMS 对称分频带探针**，并观测 Decoder 的每个内部计算阶段、1×1 两路贡献和源图候选边缘传递。

## Windows CMD

    cd /d "D:\#imageInfusion\CDDFuse"
    git pull --ff-only origin main
    .venv\Scripts\python.exe -m unittest -v test_fcsr_decoder
    .venv\Scripts\python.exe diagnose_fcsr_decoder.py

完整 val64：

    .venv\Scripts\python.exe diagnose_fcsr_decoder.py --val-scenes 64 --preview-count 8 --output-dir test_result\e2_fcsr_decoder_diagnosis_v1\val64_seed42

## 输出（默认新目录）

test_result/e2_fcsr_decoder_diagnosis_v1/val16_seed42/

- frequency_response.csv：每图×分支×频带的四个 Decoder 阶段响应；
- channel_mix.csv：1×1 两路贡献、余弦与输出饱和度；
- edge_candidates.csv：源候选边缘与单源重建、融合图、光学方向插值响应；
- summary.json：场景、checkpoint 与脚本哈希、关键均值；
- previews/*.png：源图、融合图与单源重建。

## 推进标准

高频从 Base 输出、低频从 Detail 输出，只能证明当前 Decoder 存在跨频带耦合，不能证明这种耦合有害。候选边缘在源重建中保留而在融合图中弱化，也不能区分融合器信息丢失与 Decoder 非线性重组。

若光学特征小幅插值可恢复边缘，优先排查融合后的结构供给与训练目标。仅在冻结诊断发现稳定且有害的 Decoder 频带失衡，并经真实结构核验后，才考虑等参数量可训练 FCSR 对照。最终仍必须使用匹配 val64 的 QAB/F、NAB/F、FMI、VIF、SSIM 等图像级指标。禁止将频带能量变化直接视为质量提升。
