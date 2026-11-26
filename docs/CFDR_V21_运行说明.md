# CFDR-V2.1 全量实验

CF0 采用已有光学 Base 锚定 LWPD E2。CF1 在平滑的 E2 解码预测上加入按模态增益加权的原位中心化细节。CF2 增加 8 邻域动态核和聚合强度，单模态原位权重至少为 0.5。条件网络用 1×1 卷积将 `[H_dec, D_F]` 映射至 32 通道，再以 3×3 逐通道卷积提取邻域条件。CF1 参数量为 1,006,379，CF2 为 1,006,973，差值为邻域和聚合强度预测头的 594 个参数。

两组从同一份 CF0 第 4 轮单源重建权重接续，恢复历史 AdamW 和 FP16 scaler。共享融合主干及两路单源重建均保留。新头使用融合组学习率 `1e-4`，其余学习率按 CF0 原 Stage II 配置。融合阶段为 12 轮，使用 train8000、val1000、种子 42、batch 2、梯度累积 4 和原 `L_I + 10 L_G + 0.1 L_R`。最终融合输出直接回传现有损失，按统一评价器在输出图像时裁剪及量化。

旧 CF0 checkpoint 记录了权重、优化器和 scaler，未保存数据采样器状态。CF1 与 CF2 使用相同的逐轮种子和数据顺序，CF0 对照来自其原有训练轨迹。CF2 的初始化中心权重约 0.90，初始模态增益约为 `(0.475, 0.475, 0.05)`。

## Windows CMD

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "CFDR_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%CFDR_PY%" -u run_cfdr.py smoke --device cuda
"%CFDR_PY%" -u run_cfdr.py train --device cuda
"%CFDR_PY%" -u run_cfdr.py val --device cuda --fusion-epochs 4 8 12
```

`train` 默认依次训练 CF1 和 CF2。可指定 `--arms CF1` 或 `--arms CF2`。中断后重执行同一命令会接续最近已保存的 epoch。Stage II 的第 4、8、12 轮分别对应 `epoch_008.pth`、`epoch_012.pth` 和 `epoch_016.pth`。训练日志位于 `models\e2_cfdr_v21\seed42`。

最终 val1000 复用已有 CF0 逐图指标。中间两轮使用各自 CF0 已有权重补做 val1000 推理。每个节点在单独比较组报告 14 项全图指标、各指标共同有效的场景数及 `CF1−CF0`、`CF2−CF1`、`CF2−CF0` 的 95% 配对区间。结果位于 `test_result\e2_cfdr_v21\val1000`。

全量验证完成后，用 `run_cfdr.py test --device cuda` 评测固定第 12 轮 Stage II 的 test1000。已有 CF0 归档通过协议和图像完整性检查时直接复用，CF1、CF2 保存全部三类图像。正式测试结果使用独立比较组。
