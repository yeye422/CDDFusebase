# FSC-Mamba 局部 LL 状态刷新：机制修正与匹配验证

状态：**算子与诊断已实现；尚未执行 QXS 真实图诊断或新模型训练。**
本轮不覆盖 `e2_fsc_mamba.py` 及其旧输出；新增实验采用单独的模型、评测与诊断目录。

## 依据

2026-11-17 冻结 E2 的单次 LL 初始状态诊断中，128 步扫描的后段/前段影响仅约 3.4%–3.8%。提高初态强度只放大响应，不改变相对衰减。由于 SSM 随机初始化，该结果是可达性机制证据，不是训练后融合质量结论。

## 改动

将 128 步行列双向 SSM 扫描按 16 步分段。每段从**该段对应的 LL 采样**计算状态摘要，并按以下固定规则在边界处更新：

\[
h_{j,0}=0.5h_{j-1,T}+0.5P(C_{LL,j}), \quad j>0.
\]

第一段直接使用 LL 初态，后续每段保留上一段的一半状态；中间每个 token 仍执行原始 SelectiveScan 转移，不加入新的路由、专家或训练损失。反方向的 LL 段也同步反转，避免因序列方向变化错配上下文。

三种变体均复用原 `LLStateTransfer` 的 `core`、`ll_projection`、`decay_logit`，参数量完全相同：

| 变体 | 对照目的 |
|---|---|
| `local_refresh` | 每16步刷新 SSM 状态，继承前段状态 |
| `local_feature` | 同一组 LL 分段摘要持续注入 token 特征，不刷新状态 |
| `local_shuffled` | 跨行/列错配 LL 段位置，仍执行局部状态刷新 |

原 `FSCMambaE2("state")` 单次注入可在冻结诊断中并列作为对照，旧实现和权重不修改。

## 本地 CMD：第一步（无需训练）

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
git pull --ff-only origin main
.venv\Scripts\python.exe -m unittest -v test_fsc_local_refresh
.venv\Scripts\python.exe run_e2_fsc_local_refresh.py smoke
.venv\Scripts\python.exe run_e2_fsc_local_refresh.py smoke --full
.venv\Scripts\python.exe diagnose_fsc_local_refresh.py
```

诊断使用已归档的光学 Base E2 checkpoint，只随机初始化新 SSM，比较 `single_state`、`local_feature` 和 `local_refresh` 在16、32步分段时的正确/错配/零 LL 响应。默认场景211，输出到：

`test_result/e2_fsc_local_refresh_diagnosis_v1/scene_211_seed42/`

- `profiles.csv`：行列正反向、前中后段 RMS 和正确/错配差异；
- `spatial.csv`：完整双向二维 SSM 的对照；
- `summary.json`：预训练 E2 及代码追溯信息。

所有诊断仅对内部随机初始化 SSM 的条件响应作比较，不产生融合性能结论。默认拒绝覆盖已有输出，重跑用 `--output-dir` 新路径。

## 第二步：通过诊断后再运行匹配训练

要求 `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth` 和
`run_e2_phase_guide.py train --mode feature` 对应的纯特征 SSM checkpoint 已存在；没有纯特征模型时先训练。

```cmd
.venv\Scripts\python.exe run_e2_fsc_local_refresh.py train --mode local_feature
.venv\Scripts\python.exe run_e2_fsc_local_refresh.py train --mode local_refresh
.venv\Scripts\python.exe run_e2_fsc_local_refresh.py train --mode local_shuffled
.venv\Scripts\python.exe run_e2_fsc_local_refresh.py val
```

三组复用同一 QXS train1024/val64、seed42、4轮单模态重建+12轮融合、原始损失；无参数量差异。训练/评测输出分别使用 `models/e2_fsc_local_refresh_v1_<mode>_pilot/` 和 `test_result/e2_fsc_local_refresh_v1/` 的独立目录，并校验数据、配置、代码哈希；不覆盖旧结果。完整评测依赖纯特征 SSM checkpoint，保留归档 E2 对照。重点比较 `local_refresh` vs `local_feature` 和 `local_shuffled` 的 FMI、QAB/F、NAB/F、VIF、SCD、SSIM、MS-SSIM；在无 NAB/F 恶化的前提下才考虑扩大样本、多种子确认。

## 判断边界

- 分段刷新有较强后段敏感性，不代表它更好地区分有效结构与散斑；
- 正确 LL 与错配 LL 的差异必须与零 LL 的响应量级同时报告；
- `local_refresh` 只有优于同参数量的 `local_feature` 和错配条件，才能声称“状态耦合”具有独立增量；
- 小样本 val64 只是候选筛选，不进入正式 test1000；
- 当前采用固定 chunk=16、carry=0.5，不通过验证集扫描大量超参数选择最优值。
