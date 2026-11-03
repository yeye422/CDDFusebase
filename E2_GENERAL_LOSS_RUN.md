# E2通用损失训练与验证命令

评价指标保持原paper版本。已实现结构损失与训练参数；完整训练和val1000仅由用户在CMD执行，不由代理运行。所有模式均不访问test1000。

## 推荐：匹配基线与结构项一起训练验证

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_general_loss.cmd train-val pair
```

依次训练B（gradient=10、structure=0）和S（gradient=10、structure=0.5），都从相同R3初始化训练16轮，执行固定val1000，保存全部灰度/RGB/三联预览及14项指标，自动登记训练/验证结果、校验并生成实验索引，再输出S对B的逐图配对比较。新总损失不与旧总损失直接比较。

## 分开执行

```cmd
run_e2_general_loss.cmd train pair
run_e2_general_loss.cmd val pair
```

只运行某一条件：

```cmd
run_e2_general_loss.cmd train-val s
```

但只有匹配B的val1000已完成，才生成S对B比较。默认省略条件也是pair。已有原E2可作为历史参考，本脚本默认重训匹配B以避免运行环境和代码差异影响结论。

## 第二轮可选消融

```cmd
run_e2_general_loss.cmd train-val g
run_e2_general_loss.cmd train-val gs
```

G为gradient=5、structure=0；GS为gradient=5、structure=0.5。不建议首轮直接扩大搜索。若明确要执行所有四组：

```cmd
run_e2_general_loss.cmd train-val all
```

## 工程检查

```cmd
run_e2_general_loss.cmd check
run_e2_general_loss.cmd smoke
```

check运行7项结构损失单元测试。smoke只读取一张验证图完成前向和反传，不更新优化器；已存在该单样本结果时保留并校验登记，不再次推理。其结果不代表性能。首次检查已完成，后续无需重复执行。

## 断点续训与文件保护

中断后重复原train或train-val命令，会寻找该条件下最新epoch checkpoint续训。已有epoch_016.pth则跳过训练；已有完整验证结果会核对checkpoint、切分、指标和输出策略后复用。

`workflow_protocol.json`锁定源码、初始化、切分及命令。源码或协议改变时拒绝复用旧版本，需建立新输出版本。验证中断留下不完整目录时保留旧目录并报错，不自动覆盖。当前续训沿用原训练器的epoch级恢复机制，不承诺与从未中断的训练逐位一致。

## 结果位置

```text
models/e2_general_loss_v1/b_seed42/
models/e2_general_loss_v1/s_seed42/
test_result/e2_general_loss_v1/val1000/b_seed42/metrics.csv
test_result/e2_general_loss_v1/val1000/s_seed42/metrics.csv
test_result/e2_general_loss_v1/val1000/s_seed42/comparison_vs_b/paired_comparison.csv
test_result/e2_general_loss_v1/val1000/s_seed42/comparison_vs_b/SUMMARY.md
```

SUMMARY中的门槛仅作单种子均值工程筛选，视觉检查和多种子复核仍需单独完成。脚本不提供test模式。
