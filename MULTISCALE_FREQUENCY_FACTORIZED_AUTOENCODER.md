# 光学–SAR 多尺度频率因子化对称自编码器

版本：设计稿 1.0  
状态：架构提案，尚未实现  
简称：MFFA（Multi-scale Frequency-Factorized Autoencoder）  

## 1. 核心原则

本方案只建立一个架构假设：

> 光学与 SAR 的低频主体结构具有较强共享性，而中高频信息具有明显模态差异；网络的参数
> 共享关系应由频率职责决定，而不是在同一主干中不断加入模态补丁。

因此，MFFA 不采用双 Stem、模态专用 LayerNorm、Adapter、模态门控或多个输出头。整个
Encoder–Decoder 被设计为一套可解释、对称的多尺度频率分析—合成系统：

- 低频只进入共享 Base Encoder；
- Optical/SAR 中高频进入各自的 Detail Encoder；
- 两种模态的特征最终进入同一套共享 Decoder；
- Phase 1 重建与 Phase 2 融合使用完全相同的合成路径。

## 2. 为什么需要重构现有架构

当前 Encoder 将原始单通道图像直接映射为 64 通道，再用同一个 Restormer 主干同时提取
Base 和 Detail。它没有保证 Base 一定负责低频，也没有保证 Detail 一定负责某个明确频带。
Optical 纹理、SAR 结构和 SAR 斑点可能进入相同的 Detail 表征。

当前 Decoder 将 Base/Detail 拼接后统一解码，并把原始输入图像直接加回输出。Phase 2
传入 Optical 作为旁路，因此融合输出天然带有 Optical 偏置。

MFFA 不在这些结构上追加特殊分支，而是重建完整的数据流，使每一个潜变量都有明确的频率
职责，并由对称 Decoder 验证该职责。

## 3. 总体结构

```text
                        ┌─ Shared Base Encoder ───────────── B
                        │
X ─ Laplacian Analysis ─┼─ Modality Detail Encoder (mid) ─ D2
                        │
                        └─ Modality Detail Encoder (fine) ─D1

B  ─ Shared Base Decoder ───────────────────────────────── L2_hat
D2 ─ Shared Mid Decoder  ─────────────── H2_hat ─┐
D1 ─ Shared Fine Decoder ─────────────── H1_hat ─┼─ Laplacian Synthesis ─ X_hat
```

其中：

- `B`：最粗尺度、低频主体特征；
- `D2`：中频轮廓和较大尺度边缘；
- `D1`：细粒度纹理、细边缘以及 SAR 高频变化；
- Optical 与 SAR 只在 `D1/D2` 的 Encoder 参数上不同；
- 所有 Decoder 参数共享，使两种模态和融合特征进入统一的图像合成空间。

## 4. 固定多尺度频率分析

### 4.1 两级拉普拉斯金字塔

对归一化输入 `X ∈ [0,1]` 构造：

```text
L1 = GaussianBlur(X, sigma=1.0)
H1 = X - L1

L2 = Down2(GaussianBlur(L1, sigma=1.0))
H2 = L1 - Up2(L2)
```

得到：

```text
L2: [B,1,H/2,W/2]   最粗低频
H2: [B,1,H,W]       中频残差
H1: [B,1,H,W]       细频残差
```

`GaussianBlur`、`Down2` 和 `Up2` 均使用固定算子，不包含可学习参数。边界统一使用反射填充。
第一版只使用两级金字塔，不继续增加尺度。

固定分析算子的作用不是声称频率等于语义，而是给 Base/Detail 提供稳定、可复现的职责边界，
避免网络任意交换两类特征。

### 4.2 SAR 有效区

SAR 无效像素的 mask 同步进入金字塔：

- 低频分量使用 mask 归一化卷积，避免无效零值污染邻域；
- 中高频重建损失只在对应的有效区计算；
- 不把无效区边界当作 SAR 高频结构。

## 5. Encoder

### 5.1 Shared Base Encoder

Optical 和 SAR 的 `L2` 共用同一套 Base Encoder：

```text
L2
→ Conv3×3, 1→48
→ 4 × Restormer Block, dim=48
→ Conv3×3, 48→64
→ B [B,64,H/2,W/2]
```

Base Encoder 不接收模态标识，不使用模态专用归一化。网络必须用相同的变换处理两种模态的
低频图像，从结构上建立共享空间。

### 5.2 Optical Detail Encoder

Optical 的 `H2/H1` 由一套层级 Detail Encoder 处理：

```text
H2 → Conv3×3 → 2 × Restormer Block ───────────────→ D2_o

H1 → Conv3×3 → 2 × Local Detail Block
                  ↑ Up2(project(D2_o))
                跨尺度残差连接
                                                   → D1_o
```

Local Detail Block 使用：

```text
Depthwise Conv3×3 → GELU → Pointwise Conv1×1 → residual
```

中频路径提供较大尺度上下文，细频路径保留 Optical 的纹理和精细边缘。

### 5.3 SAR Detail Encoder

SAR 使用与 Optical Detail Encoder 完全相同的拓扑，但参数独立：

```text
(H2_s, H1_s) → SAR Detail Encoder → (D2_s, D1_s)
```

SAR 分支不增加去斑头、可靠性门控或 Speckle 分类器。不同参数使它能够形成适合 SAR
中高频分布的表示；是否抑制无用斑点由最终融合目标和跨尺度一致性验证，而不是由未经监督的
分支名称决定。

### 5.4 Encoder 输出接口

统一接口：

```text
B, D2, D1 = Encoder(X, modality)
```

`modality` 只选择 Optical 或 SAR Detail Encoder，不影响 Base Encoder。

## 6. 对称 Decoder

Decoder 不接收原始 Optical 或 SAR 图像，也不接收模态标识。

### 6.1 Base Decoder

```text
B
→ Conv3×3, 64→48
→ 4 × Restormer Block
→ Conv3×3, 48→1
→ L2_hat
```

### 6.2 Mid Decoder

```text
D2
→ 2 × Restormer Block
→ Conv3×3, 64→1
→ H2_hat
```

### 6.3 Fine Decoder

```text
D1
→ 2 × Local Detail Block
→ Conv3×3, 64→1
→ H1_hat
```

### 6.4 拉普拉斯合成

```text
L1_hat = Up2(L2_hat) + H2_hat
X_logit = L1_hat + H1_hat
X_hat = sigmoid(X_logit)
```

Decoder 对 Optical、SAR 和融合图完全共享。不存在：

- Optical/SAR/Fusion 三套输出头；
- 模态条件归一化；
- 原始图像旁路；
- 为 Phase 2 单独新增的随机 Decoder 参数。

## 7. Phase 1：频带可验证的自编码训练

### 7.1 前向路径

```text
Optical:
  (L2_o,H2_o,H1_o) → (B_o,D2_o,D1_o)
                    → (L2_o_hat,H2_o_hat,H1_o_hat)
                    → O_hat

SAR:
  (L2_s,H2_s,H1_s) → (B_s,D2_s,D1_s)
                    → (L2_s_hat,H2_s_hat,H1_s_hat)
                    → S_hat
```

### 7.2 损失

Phase 1 使用一套统一损失：

```text
L_phase1 = L_image + lambda_band * L_band + lambda_dec * L_decomp
```

其中：

```text
L_image = L1(O_hat,O) + DSSIM(O_hat,O)
        + masked_L1(S_hat,S) + masked_DSSIM(S_hat,S)

L_band = L1(L2_o_hat,L2_o)
       + L1(H2_o_hat,H2_o)
       + L1(H1_o_hat,H1_o)
       + masked counterparts for SAR
```

`L_decomp` 只约束同一模态内 Base 与 Detail 的冗余，不强迫 Optical/SAR 的 Detail 对齐。

第一版建议：

```text
lambda_band = 1.0
lambda_dec = 0.1
```

不使用单独的 Optical 梯度损失，因为 `H1/H2` 的显式频带重建已经承担边缘监督。若继续保留
旧梯度损失，会重复强调高频并破坏架构归因。

## 8. Phase 2：同构的多尺度融合

### 8.1 分尺度融合

```text
B_f  = BaseFuse(B_o, B_s)
D2_f = MidFuse(D2_o, D2_s)
D1_f = FineFuse(D1_o, D1_s)
```

三个 Fuse 使用相同的残差融合单元拓扑，但参数不共享，因为三个频带承担不同职责。

建议融合单元：

```text
concat(F_o,F_s)
→ Conv1×1, 128→64
→ 2 × Restormer/Local Block
→ residual projection
```

- BaseFuse 使用 Restormer Block；
- MidFuse 使用 Restormer Block；
- FineFuse 使用 Local Detail Block，避免在最细频带引入过强全局混合。

### 8.2 统一合成

```text
F = Decoder(B_f,D2_f,D1_f)
```

Phase 2 不改变 Decoder 结构，也不引入 Optical 输入旁路。

### 8.3 C0-minimax

现有 frequency-minimax思想可以迁移到三个明确频带：

```text
J_low  = Optical低频保真目标
J_mid  = SAR中频结构目标
J_fine = Optical细节保持 + SAR高频上限约束

L_C0 = smooth_max(
    J_low / J_low_ref,
    J_mid / J_mid_ref,
    J_fine / J_fine_ref
)
```

第一轮若要严格判断架构收益，也可以先保持现有最终图像C0-minimax损失不变；三频带minimax
作为后续损失改进，不与首次架构对比同时引入。

## 9. 初始化策略

MFFA 是新架构，不追求与旧 B0 的逐像素等价初始化。

允许迁移的只有同构基础块：

- Shared Base Encoder 的 Restormer Block 可复制 IVF Encoder block；
- Optical/SAR Detail Encoder 的 Restormer block 均复制同一份 IVF block作为起点；
- Shared Decoder 的 Restormer block可复制IVF Decoder block；
- 新增输入投影、频带投影和Fuse使用标准初始化。

该迁移仅用于优化稳定性，checkpoint必须标记为新架构，不能伪装成旧B0或C0。不得使用
`strict=False` 静默忽略键；必须使用显式迁移函数并记录来源哈希。

## 10. 相比补丁式方案的结构优势

| 补丁式设计 | MFFA |
|---|---|
| 原图进入同一Stem后再区分模态 | 输入先被统一频率分析 |
| 双Stem处理所有频率 | 只对模态特异的中高频分开编码 |
| 模态LayerNorm散布在每个Block | 完全删除条件归一化 |
| 双Detail Head只是参数复制 | Detail Encoder具有明确频带输入与职责 |
| Decoder需要Optical/SAR/Fusion mode | Decoder完全无模态条件 |
| Base-derived Skip是局部补偿 | Base Decoder是完整低频合成路径 |
| 多个修改点分别解释 | 所有模块由同一个分析—合成原则推出 |

## 11. 参数与算力边界

建议目标：

- 总参数量控制在 1.8M以内；
- 输入 patch仍为128×128；
- 低频Base在半分辨率运行，抵消新增Detail Encoder的开销；
- Fine路径只使用局部块，不在全分辨率增加通道自注意力；
- 单图显存不超过当前模型的1.5倍；
- 保持输出单通道，颜色重建继续使用Optical色度。

## 12. 两组直接比较

第一轮只保留两个最终模型：

### G0：C0-Original

```text
models/reliability_v5/frequency_minimax_baseline_pilot/epoch_005.pth
```

### G1：MFFA-C0

```text
MFFA Phase 1
→ MFFA Phase 2
→ 相同train800/val200、5 epoch、seed42的C0-minimax
→ 固定epoch_005.pth
```

两组在相同val200上直接比较。该比较回答“完整MFFA系统是否优于当前C0”，不单独归因某个
频带或模块。若需要发表严格的架构归因，再在G1通过后补充消融，不在第一轮拆成多个步骤。

## 13. 通过条件

MFFA-C0必须同时满足：

- QAB/F不低于C0-Original；
- FMI和VIF不出现系统性下降；
- NAB/F下降不能以QAB/F、FMI或VIF下降为代价；
- SSIM与MS-SSIM至少保持；
- 建筑、道路和田块边界保持连续；
- SAR密集斑点与孤立伪边缘不增加；
- 输出不坍缩为Optical复制；
- 逐图改善比例具有一致性，而不是由少量异常图贡献；
- 最终结论在未参与选择的test split确认。

## 14. 实施边界

正式实现时应作为新的独立架构文件接入，不修改原 `net.py` 中的CDDFuse类。原始训练和推理
路径必须保持可复现。新架构使用独立variant/checkpoint键、独立训练脚本和独立结果目录。

在代码实现前先冻结以下内容：

1. 两级拉普拉斯分析与合成公式；
2. Base/Detail Encoder通道数和Block数量；
3. Phase 1损失及权重；
4. Phase 2三个Fuse的统一接口；
5. C0首次对比是否保持旧损失不变。

未经上述冻结，不开始正式训练，避免再次演变为边训练边追加局部模块。
