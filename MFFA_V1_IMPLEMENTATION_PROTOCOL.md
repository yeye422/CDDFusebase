# MFFA v1 第一版实现协议

## 已实现

- logit域固定两级频率分析与严格可逆合成；
- Optical/SAR共享Low Encoder；
- Optical/SAR参数独立的Mid/High Encoder；
- 来源保持Low/Mid融合；
- Optical高频锚定、SAR 5% RMS有界残差；
- 不含原始图像旁路的共享对称Decoder；
- Phase 1图像与三频带联合重建训练器；
- 单元测试与dry-run入口。

## 当前阶段

只执行Phase 1自编码。Low/Mid/High Fusion模块在Phase 1冻结，Phase 1通过工程与重建闸门后才实现Phase 2融合训练。

训练命令由用户在Windows CMD中执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_mffa_v1_phase1_seed42.cmd
```

首次AMP运行在创建目录后因SSIM输入dtype不一致中止，遗留的`models\mffa_v1\phase1_seed42`只包含配置和空历史，不覆盖也不作为实验结果。修复后正式运行使用唯一目录`models\mffa_v1\phase1_dtypefix_seed42`。

Phase 1完成前禁止运行融合test1000。

Phase 1训练完成后执行独立重建评价：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_mffa_v1_phase1_validation.cmd
```

评价分别报告Optical/SAR的MAE、MSE、PSNR、SSIM和Low/Mid/High频带归一化误差，并保存固定8张`源图/重建/绝对差值×4`预览。只有两种模态都能稳定重建且High频带不塌缩，才进入Phase 2融合训练。

## Phase 2

Phase 1已通过后，第一轮Phase 2只训练`low_fusion`、`mid_fusion`和`high_fusion`，其余模块全部冻结。使用与C0相同的frequency-minimax损失、训练参考、5 epoch、学习率和梯度裁剪，不加入SPD、Gate或新损失。

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_mffa_v1_phase2_seed42.cmd
run_mffa_v1_phase2_validation.cmd
```

validation保存B0/MFFA完整14项指标和固定8张`Optical/SAR/B0/MFFA`预览，同时保存MFFA解码Low/Mid/High频带图。未通过validation前禁止执行正式test1000。

首次Phase 2评价在创建输出目录后因MFFA输出Tensor保留计算图而中止。评价器已增加全局`no_grad`和显式`detach`；修复版使用唯一目录`test_result\mffa_v1\phase2_seed42_val1000_evalfix`，不覆盖失败目录。
