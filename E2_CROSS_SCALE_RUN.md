# E2 跨尺度条件小试：实现与运行协议

日期：2026-11-19。本轮落实上一轮A3：先检验跨尺度条件＋单一融合器，不加入MoE、低秩分解、信息瓶颈或新损失。

**当前状态：代码、控制组、训练/验证入口已完成。19项CPU合成测试通过。启动预检已补充CUDA构建及设备可用性检查。没有运行真实数据训练或批量验证，尚无新画质收益结论。**

**模型与接口。**

基底为光学Base保留E2。共享Encoder、LWPD、`Bo+C(Bo,Bs)`、原三条膨胀卷积分支、原聚合层和Decoder均保留。两个来源共享新增条件网络的参数，但分别从各自Detail计算条件，不在新增条件支中交换另一源value。LWPD和原融合仍有跨源交互，不能称整网来源隔离。

条件支重新对LWPD重建后的Detail取同网格一级Haar，使用LH/HL/HH绝对值组成通道归一化描述，再投影为8通道。条件描述在1、1/2、1/4网格形成金字塔，统一上采样后进入条件头。原位有符号Detail内容不走这条池化路径，原64→32来源投影继续提供成图内容。

关系版增加三个尺度对的通道余弦一致性统计；这只是可学习描述的一致性，不是几何真值、质量概率或SAR散斑判别标签。

每源32通道投影使用 `z' = z × (1+δ)`，其中 `δ=0.1×tanh(condition)`，再进入原聚合。投影保留系数位于0.9–1.1，避免硬删除来源。它不保证图像域没有过冲，也不能把系数解释为来源贡献率。

条件头末层零初始化，保证与**同次随机初始化的原接口**一致；不是加载已有E2权重复现已训练输出。新增网络2,536参数，三种模式完全同形；普通上下文与关系版具有相同描述金字塔和主卷积MAC，但非卷积统计运算不同，尚未测总FLOPs和耗时。

实现：[模型](e2_cross_scale.py)、[流程](run_e2_cross_scale.py)、[测试](tests/test_e2_cross_scale.py)。

**控制组。**

| 模式/条件 | 内容 | 可以检验什么 |
|---|---|---|
| `relation` | 尺度描述＋三组跨尺度余弦统计 | 首个候选 |
| `wide` | 相同尺度描述＋各尺度普通通道均值统计 | 显式关系统计是否超出普通多尺度宽上下文 |
| `shuffled` | 与relation同结构，但训练时将全分辨率调制图确定性滚动半幅图像 | 可选的独立训练错位条件控制 |
| 同权重 `shifted` | 验证时滚动调制图，精确保留调制值分布 | 输出是否依赖正确条件位置 |
| 同权重 `disabled` | 验证时将调制置零 | 已训练模型的即时条件作用 |

`wide`并非完全没有尺度信息，而是没有显式的尺度对余弦统计。`shifted/disabled`同权重诊断不能代替独立训练消融，也不能只靠它们的退化证明关系机制有质量收益。关系候选优于归档E2仍不足以归因，需与`wide`比较；有信号后再决定是否补独立训练`shuffled`。

**训练协议。**

沿用已归档原损失pilot的train1024/均匀val64清单、seed42、batch2/累积4、4轮源图重建＋12轮融合。新候选整套从头初始化，重建阶段冻结融合层，融合阶段沿用原E2强度＋10倍梯度＋0.1源重建损失；没有额外关系标签、基础模型或测试集使用。

第一轮只运行新增`relation`，不重训基线。原损失E2权重仅用于证明对照身份，比较默认复用`test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/e2`中的既有指标和灰度图。脚本先核对checkpoint、manifest、场景顺序、评价口径和图像文件哈希；不满足时直接报错，不静默切换数据或重训。

评测固定第16轮，使用 `reference_v2_candidate_with_paper_fmi` 和clamp/round/uint8。每个新模型输出aligned、shifted、disabled三个读出及归档E2对照，保存完整14项逐图结果、每项共同有限场景、均值/标准差和配对区间，并保存64张灰度、RGB、三联预览。RGB仅保留光学色度，指标使用灰度亮度。

附加读出包括两源SSIM/MS-SSIM及源图梯度候选的有符号传递，后者统一在量化图上计算，候选不是真值。没有候选的场景保留非有限值并逐项按共同有限场景处理，不能与旧浮点探针直接混用数值。

**Windows CMD。**

检查CUDA、归档和配置，不训练、不进行模型推理：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_e2_cross_scale.cmd preflight
```

首个候选的训练＋val64，由用户执行：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_e2_cross_scale.cmd
```

这等同于`train-val --mode relation`，只训练一个新候选。若中断，恢复同一次训练：

```cmd
run_e2_cross_scale.cmd train-val --mode relation --resume
```

续训保留本次模型、优化器、AMP及采样/增强随机状态；这不是用历史E2初始化。若尚未生成第一个checkpoint则不能续训，应保留旧目录并使用新tag。

候选有信号后，补普通宽上下文并形成同组比较：

```cmd
run_e2_cross_scale.cmd train-val --mode wide
run_e2_cross_scale.cmd compare
```

可选的独立训练错位控制：

```cmd
run_e2_cross_scale.cmd train-val --mode shuffled
run_e2_cross_scale.cmd compare --include-shuffled
```

已完成的训练和验证会在校验后复用。`compare`仅读取已有结果，不做模型推理，并重新按所选方法的共同有限场景汇总；基线也登记在同一个新比较组，避免跨组直接排列。训练完成及评价/比较完成后自动更新注册表并执行`maintain_experiment_index.py --check --render`。

如改变源码或参数，已有训练的工作流锁会拒绝复用/覆盖。需要明确的新实验时使用新tag，例如`--tag v2`，并在该tag下运行各控制；不删除历史目录，也不沿用不匹配的旧训练。如果之前仅生成流程锁、连seed42训练目录都尚未创建，则重试前自动保留旧锁为`workflow_protocol.prestart_<hash>.json`并记录新锁；已有训练目录时绝不自动刷新。评测记录源码、配置、输入、清单、checkpoint及结果文件哈希，并核对运行库版本。

**CUDA 环境。**

本机项目 `.venv` 安装了 `2.4.1+cpu`，启动器默认使用已验证的 `D:\anaconda\envs\lightglue_env\python.exe`（PyTorch `2.5.1+cu121`，RTX 4060 Ti）。启动器会打印实际使用的解释器。若要用其他环境，在同一个 CMD 窗口用 `set "E2_PYTHON=绝对路径\python.exe"` 指定解释器后运行 `preflight`。若本机默认环境路径不存在且未指定 `E2_PYTHON`，启动器会使用项目 `.venv`，训练前需确认其中的 PyTorch 支持 CUDA。

如果此前的 CUDA 错误仅留下 `workflow_protocol.json`，还没有 `seed42` 目录，预检通过后直接运行默认命令。启动时会保留旧协议快照并写入本次协议。

**输出。**

- 模型：`models/e2_cross_scale_pilot_v1_relation/seed42/`，含config、manifest、history和逐轮checkpoint；父目录有`workflow_protocol.json`。
- 验证：`test_result/e2_cross_scale_pilot_v1_relation/val64/seed42/`，含e2/aligned/shifted/disabled图像与指标、`paired_comparison.csv`、排除场景清单和运行元数据。
- relation/wide最终同组表：`test_result/e2_cross_scale_comparison_v1/relation_wide_val64_seed42/`。这里通过metadata引用各原始验证图像，不复制或重新生成模型输出。

wide/shuffled替换路径中的模式名；新tag替换v1。流程不提供test1000入口，避免开发阶段自动进入测试集。

**查看结果后的判断。**

优先检查困难光学结构是否有可辨认改善，同时保住SAR独有边/孤立目标；NAB/F以不劣于匹配E2为筛查目标，联合看QAB/F、FMI、VIF、SSIM、MS-SSIM、SCD及完整指标。若需非劣容差，应在看结果前确定，不把差异不显著解释为等效。

如果只改善调制统计或源恢复探针，没有最终成图收益，先停止；如果wide同样有效，不主张显式关系成立；若NAB/F改善来自删掉SAR有效内容，也不进入MoE。val64是反复使用的开发集，配对区间不代表训练种子或地理独立验证。

**本轮验证范围。**

19项合成测试覆盖有符号Haar入口、初始等价、奇数/极小尺寸、条件位移、参数匹配、两阶段真实损失与优化器分组、梯度、整网重载、同次checkpoint及随机状态恢复、共同有限统计、CUDA预检分支和首次启动失败后的锁保留。训练吞吐和真实图像融合收益仍须由后续CMD执行结果提供。
