# E2-G1：PA 引导的方向梯度损失

## CMD 训练命令

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_g1.cmd train
```

训练后执行固定 val1000，并自动登记结果和生成配对 Bootstrap 报告：

```cmd
run_e2_g1.cmd val
```

也可一次执行两个阶段：`run_e2_g1.cmd train-val`。这些命令仅供用户执行，开发过程没有启动训练或批量验证。默认只运行 G1，不重训 B/S。初始化为原 R3，而非已训练 B；seed42、2+4+10轮、batch2、累积4步、学习率与现有匹配 B 一致。运行前核验 B 的配置和初始化哈希。

## 实现约定

网络保持 lwpd_e2，无 PAC-Mamba。总损失为 `intensity + 10 * pa_directional_gradient + 0.1 * reconstruction`，不新增融合 SSIM、IB 或对比损失；原有重建损失保持不变。

- 固定 Monogenic PA，从原图独立计算，不读取网络门控；波长4/8/16，反射填充16。
- 置信度为跨尺度 PA 均值经 `(C-0.3)/0.7` 截断到[0,1]；光学与 SAR 置信度乘以 `max(0, cos(2*(thetaO-thetaS)))`，3×3平滑后乘 SAR 有效区域的3×3腐蚀掩码。证据与方向均 StopGrad。
- 使用光学 PA 法向，独立有符号 Sobel Gx/Gy；坐标 x 向右、y 向下，与 PA 一致。模态对比度可能反转，因此匹配投影绝对值。
- 共同区域法向目标为 `min(max(abs(On),abs(Sn)), abs(On)+0.25)`，限制原始 SAR 梯度增量；切向匹配光学，权重0.25。Sobel不除以8，截断值0.25指该响应单位。
- Smooth L1（Huber形式，beta=0.1）；最终梯度项为 `mean((1-M)*光学幅值约束 + M*(法向约束+0.25*切向约束))`。低置信度及 SAR 独有区域不强制匹配 SAR 梯度。按全图均值聚合，避免少数可信像素被归一化放大。
- SAR 无效值在 FFT 前用光学填充，避免无效数值污染全图频域证据；无效 Sobel 邻域禁用 SAR 方向监督。该填充仍可能影响邻近证据，不等价于严格的掩码傅里叶变换。

这些阈值是首版固定工程超参数，PA 不是经过校准的结构真值，效果需训练验证。

## 输出和复用

模型：`models/e2_g1_v1/g1_seed42/epoch_016.pth`。
验证：`test_result/e2_g1_v1/val1000/g1_seed42/`，保留完整灰度/RGB/三联图及原 paper 14项指标。
报告：该验证目录下 `comparison_vs_b_bootstrap_v1/SUMMARY.md`。

协议锁记录源码、初始化、切分哈希。重复训练命令自动从最新 epoch 续训，已有最终 checkpoint 则跳过。沿用原训练器 epoch 级恢复，不保证中断恢复与连续训练逐位一致。已有完整验证会核验元数据后复用，不覆盖不完整目录。训练/验证完成后自动登记 EXPERIMENT_REGISTRY.json 并校验生成索引；本次仅开发与单元测试，未新增训练或验证结果条目。

直接对照复用 `e2_general_loss_v1/b_seed42`，不混用历史 E2-B 的均值。按审查要求保留历史 NAB/F 上限0.067842，并要求 QAB/F不低于直接对照、MI/VIF改善；同一1000张验证图进行2000次配对 Bootstrap，seed42，报告候选减基线的95%百分位区间。该区间不代表训练种子方差。相同数据及评价协议沿用已有比较组；脚本核对两条注册记录的 comparable_group。

重新生成缺失比较报告可执行 `run_e2_g1.cmd compare`。不提供自动 test1000 流程。

## 工程验证

`run_e2_g1.cmd check`：12项单元测试，涵盖 PA法向及180度周期、无效 SAR/NaN隔离、证据无梯度、恒等输入、完整 E2-G1反传、原结构损失及零权重旧路径回归。开发时已通过；尚未验证训练收敛与融合质量。
