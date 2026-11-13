# LWPD LL-context 训练、验证与测试命令

日期：2026-11-04。对应[设计方案](./LWPD_LL_CONTEXT_V1_DESIGN.md)，当前已实现独立版本 `lwpd_e2_llctx_v1`。本入口只训练E2+LL-context，不包含PAC-Mamba。

结果更新：12轮训练及val1000已完成，当前候选未通过结构/伪影筛选，不建议继续执行下述test命令。详见[结果分析](./reports/lwpd_pac/LWPD_LL_CONTEXT_V1_VAL1000结果分析_20261104.md)。下列命令保留供复现，不代表推荐重复运行。

## 已完成检查

- 新分支新增9520参数，固定Haar和原高频门控公式，原损失保持 `L_I+10L_G+0.1L_R`。
- 从已有E2严格加载；只允许LL-context参数缺失，旧LWPD权重不会静默随机初始化。
- 专项检查覆盖零初始化输出/损失/代表参数梯度一致、奇数形状和增益边界、新分支梯度可达性、评价checkpoint兼容与优化器分组。
- 真实单样本FP32前向反传通过：loss=0.8553025723，所有已有梯度有限，新LL修正和LL变化均为0，无优化器更新。
- 单样本已登记 `LWPD-E2-LLCTX-V1-SEED42-SMOKE`；注册表检查通过。本次未运行完整训练、val1000或test1000。

## 推荐：训练后自动验证

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_lwpd_e2_llctx.cmd train-val
```

从已有 `models\lwpd_qxs_ablation_v1\e2_seed42\epoch_016.pth` 出发追加12轮，2 warmup+3 local+7 joint，batch2、累积4，seed42，固定train8000/val1000。新分支与旧模块采用设计中的分组学习率，不重训原E2。

结束后自动对固定最终 `epoch_012.pth` 做val1000，保存14项逐图指标、均值/标准差、全部灰度/RGB/三联图，登记结果，并与已有原E2 val1000生成配对摘要。

## 分开执行

仅训练：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_lwpd_e2_llctx.cmd train
```

仅验证已训练的最终checkpoint：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_lwpd_e2_llctx.cmd val
```

## 正式test1000

先完成val1000并查看配对结果。固定模型后，显式执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_lwpd_e2_llctx.cmd test
```

该命令要求val配对摘要已存在；不会自动按指标挑选checkpoint或反复测试。评价固定 `epoch_012.pth`，保存1000张灰度、1000张RGB、1000张三联图及完整14项逐图均值/标准差，登记test结果，更新项目规定的五方法汇总，并在新测试目录另存五方法+原E2+LL-context的补充表。

旧paper指标口径保持不变，旧/新参考指标不混表；对模型比较的含义应限定在现有协议内。

## 结果位置

| 内容 | 目录 |
|---|---|
| 训练checkpoint与history | `models/lwpd_e2_llctx_v1/seed42` |
| 源码/参数/初始化哈希锁 | `models/lwpd_e2_llctx_v1/workflow_protocol.json` |
| val1000图像及指标 | `test_result/lwpd_e2_llctx_v1/val1000/seed42` |
| val配对结果 | 上述目录的 `comparison_vs_e2/SUMMARY.md` |
| test1000图像及指标 | `test_result/lwpd_e2_llctx_v1/test1000/seed42` |
| test配对结果 | 上述目录的 `comparison_vs_e2/SUMMARY.md` |
| test补充比较表 | 上述目录的 `formal_comparison/SUMMARY.md` |

重复train会跳过已完成候选或从最后epoch继续；重复评价会核对已有完整结果，不覆盖原图。中断但无完整结果的评价目录会拒绝覆盖，必须保留旧目录并使用新版本路径。源码或协议变化也须新建版本，不能移除哈希锁强行续跑。

## 仅做工程检查

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_lwpd_e2_llctx.cmd check
```

smoke已完成，不需重复；已有smoke目录会拒绝覆盖。

追加训练与原E2比较时，收益包含额外12轮训练，不能只归因于模块。默认复用原E2，不自动增加无LL-context微调对照；严格模块归因需另立同预算对照。
