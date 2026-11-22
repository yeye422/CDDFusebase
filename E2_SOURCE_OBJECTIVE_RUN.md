# E2 源结构监督与源别读出：实验入口

日期：2026-11-19。已实现候选目标、可选源别读出、冻结单场景目标诊断及三组显式训练/验证入口。当前只有合成单元测试和清单/配置核查，没有真实数据推理、完整训练或融合收益结论。

默认只做冻结单场景目标诊断。默认训练分支为仅改监督 `objective`；不会默认训练原 E2、源别读出或联合版本。完整模型训练从头初始化，不加载 E2/R3 权重；冻结诊断才读取归档 E2。

2026-11-19 修复：Python 3.8 从 CMD 直接运行脚本时，`__file__` 可能为相对路径。现在在入口切换工作目录前固定脚本和模块的绝对路径，用于协议签名与哈希；新增根目录及父目录相对调用的回归测试。该签名异常发生在创建训练目录之前，可直接重跑原命令，无需清理结果目录。

2026-11-19 环境修复：训练前先检查 CUDA 和重建损失接口，再写协议锁。若之前的失败只留下协议锁、尚无 `seed42` 训练目录，入口会备份旧锁为 `workflow_protocol.unstarted-<hash>.json` 后刷新；已有训练目录或权重时仍拒绝更换协议。配置新增 `source_runtime`，记录解释器、PyTorch、CUDA 构建及关键依赖版本。

## 选择可用 GPU 解释器

当前项目 `.venv` 的 PyTorch 为 `2.4.1+cpu`，不能用于 CUDA 训练。本机已核查可用的 GPU 环境是 `D:\anaconda\envs\lightglue_env\python.exe`，Python3.10.19、PyTorch2.5.1+cu121，可识别 RTX4060Ti。启动器现在按以下顺序选择解释器：显式 `E2_PYTHON`、本机已验证的 `lightglue_env`（路径存在时）、项目 `.venv`。因此本机直接运行原命令即可，无需每次设置环境变量，也不安装或修改 Conda 环境。

先检查默认选择：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_e2_source_objective.cmd check
```

看到 `[E2] Python` 指向 `lightglue_env`，并显示 `CUDA ready` 与 `Reconstruction loss ready` 后，再执行下文命令。只有要使用其他环境时才设置 `E2_PYTHON`；此显式设置始终优先。训练和验证应使用同一选定环境，协议会核对记录的运行环境；本次切换不是基线重训。

该 GPU 环境使用 Kornia0.8.1，已移除旧 `losses.SSIM`。兼容层 `utils/legacy_dssim.py` 保留原 Kornia0.2 重建损失的公式、反射边界及 `clamp(1-SSIM,0,1)/2`，没有直接换成截断范围不同的新版 `SSIMLoss`。CPU环境中已对照旧实现的数值及梯度；GPU小尺寸前向/反向、AMP及14指标合成兼容检查已通过。尚未执行完整训练。

## 改动与实验边界

实现位于 `e2_source_objective.py`，原始 `DirectFusionLosses` 和原 E2 模型未改写。`test_OS.py` 仅新增两个 checkpoint variant 的注册，支持正常自动识别和推理。

| `--arm` | 模型 | 融合监督 | 默认运行 |
|---|---|---|---|
| `objective` | 与光学 Base E2 同结构、同种子初始化 | 新目标 | 是，仅在显式调用训练时 |
| `readout` | 两源较晚读出 | 原 `LI+10LG` | 否 |
| `joint` | 与 `readout` 相同结构 | 新目标 | 否 |

第四格为已有 `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。验证复用已保存的现行协议 E2 输出，来源为 `test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/e2`，执行前核对原 checkpoint、manifest、指标口径、场景顺序和训练设置；不重训、不重复推理已有对照。

三组均复用同一 train1024/val64、4轮源图重建+12轮融合、seed42、batch2、累积4及原学习率安排；清单中 test 为 0。仅改监督的模型参数结构与原 E2 完全相同。源别读出会删除原联合 Detail 头、增加源别图像残差头，参数量并未与原头严格匹配，不据此声称算子容量归因成立。

## 新监督的确切定义

融合阶段：

`Lfuse = 1.0 * Llow + 5.0 * Lpreserve + 1.0 * Luncertain`。

`Ltotal = Lfuse + 0.1 * Lreconstruction`。

这是一组固定的小试超参数，未经真实验证集筛选，不能视为推荐的最终最佳权重。原参数 `gradient_weight=10` 保留在对照配置中，但新目标使用上述显式权重，不把 `Lpreserve` 再乘10。

- `Llow`：先逐像素取有效源最大强度，再对目标和输出做2×2均值投影，相当于固定 Haar LL/2；奇数尺寸使用右/下 replicate padding。不是先低通两源再取最大值，也不是对重建 Detail 池化。
- `Lpreserve`：固定5×5条带差分核，在水平、垂直和两条对角线读取有符号局部对比；Smooth L1 的 beta 为0.02。光学、SAR、共同候选分别按自身软支持量归一化，再对非空组平均；不以两源全局幅值竞争选择唯一赢家。
- 软证据：由本源对比幅值、邻域符号一致性及轻微平滑前后的一致性产生，尺度下限0.01；全部 detach，不读取融合结果或人工标签，不由网络自由关闭。它只是启发式代理，不能称真结构、去斑置信度或物理可靠性。
- 同方向同符号的共享支持只匹配一个加权共同目标；相反极性的重叠支持进入不确定项，避免同时强迫两个冲突的有符号目标。单源额外支持仍保留，不要求另一源也有响应。
- `Luncertain`：按不确定软权重计算弱的原最大 Sobel 幅值误差，没有新增全分辨率最大强度项，也不把全部未选区域回退为光学。无效 SAR 或不完整 Sobel 邻域采用有效光学目标。该项仍可能残留 SAR 偏向，必须通过最终图像验证。
- SAR 候选采用完整7×7有效支持，防止条带核、平滑和邻域统计读取无效 SAR；固定候选与有效区域外的 SAR 数值无关。

日志中 `gradient` 对新目标表示 `Lpreserve`，不再表示旧 Sobel 损失。另保存 `legacy_intensity`、`legacy_gradient`、各组结构误差与软支持比例，不能直接把不同目标的 `train_loss` 当画质分数比较。

这些分组仍可能漏掉弱光学边或保留 SAR 干扰。当前实现不是旧 source-interval 的可靠性认证，也不保证 NAB/F 改善。

## 可选源别读出

保留原共享 Encoder、LWPD E2 及光学 Base 锚点。共享 Decoder 主干读取 Base 和零 Detail，提供上下文与 Base logits；两路32通道轻量头分别读取本源 Detail 和该上下文，生成独立受限残差，最后 sigmoid 合成一张图。

残差使用 `2*tanh(head)`，独立局部门为 sigmoid，范围0到1。两路最终残差投影零初始化，门初值0.5。源图重建阶段分别训练相应读出，融合阶段两路同时参与。新头构造使用隔离 RNG，保留后续随机采样/增强的 RNG 序列。

`readout` 与 `joint` 不再运行旧联合 Detail 压缩头；不是把一个未训练分支挂在原输出后声称已解决问题。共享 Base 和 LWPD 仍有跨源条件，两路并非信息论上的纯单源，门值也不是来源贡献率。通用单源重建必须显式指定源别，训练已接入 `reconstruct_source`。

## 运行顺序与 Windows CMD

在项目根目录执行。以下训练和真实图像诊断命令均由用户执行，代理本轮未运行。

先做冻结单场景诊断：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_e2_source_objective.cmd probe
```

默认验证场景211，读取归档 E2 的未量化浮点输出。比较旧目标和新目标的图像梯度方向，直接图像改变量严格匹配；目标 MAE 上限为0.25个8位灰度级，单像素上限4个灰度级，若方向不能达到则使用两方向共同可达预算。预算只由梯度效应决定，不用融合分数搜索强度；匹配失败会报错。

结果默认写到 `test_result/e2_source_objective_probe_v1/scene211_image`，保存旧/新目标损失、实际幅度、完整14项指标及灰度/RGB/预览。该单图结果不能自动准入训练，目标自身下降也不是结构恢复证据。

需要检查现有 Decoder 的有界局部可达性时，单独执行：

```cmd
run_e2_source_objective.cmd probe --project-detail
```

结果写到独立 `scene211_detail`。仅对临时 Detail 做最多8步有界拟合，RMS不超过原 Detail 的4%，模型参数冻结、不保存新 checkpoint。该拟合只选择更接近指定图像目标的步骤，不用画质指标选步；它与直接图像修正的实际幅度不保证相等，元数据单独记录，不能混作等幅归因。

选择其他固定 val64 场景或重算时必须使用新输出目录，例如：

```cmd
run_e2_source_objective.cmd probe --scene-id 546 --output-dir test_result\e2_source_objective_probe_v1\scene546_image_review01
```

检查独立困难边、SAR独有内容和负例后，如决定进入小试，仅训练新监督版本：

```cmd
run_e2_source_objective.cmd train-val --arm objective
```

此命令只训练一个新增模型并评价 val64，不训练基线、不访问 test。也可将 `train`、`val` 分开运行。已有完整训练会校验并复用；部分训练目录会拒绝覆盖，本入口暂未增加断点续训，不能删除历史结果后假装继续。

只有后续需要分离读出贡献时才显式运行：

```cmd
run_e2_source_objective.cmd train-val --arm readout
run_e2_source_objective.cmd train-val --arm joint
```

三个候选均已有结果时，按四方法共同有限场景重建四格表；该命令不训练或补跑缺失格：

```cmd
run_e2_source_objective.cmd compare
```

合成检查不读取真实数据或 checkpoint，也没有优化器更新：

```cmd
.venv\Scripts\python.exe -m unittest -v tests.test_e2_source_objective
run_e2_source_objective.cmd smoke --arm objective
run_e2_source_objective.cmd smoke --arm joint
```

## 结果与复用规则

- 每个候选训练目录：`models/e2_source_objective_v1_<arm>/seed42`。
- 每个候选验证目录：`test_result/e2_source_objective_v1_<arm>/val64/seed42`。
- 四格比较目录：`test_result/e2_source_objective_v1/four_cell_val64/seed42`。
- 验证保存候选全部64张灰度、RGB和三联图；逐图14指标、标准差、共同样本数与配对区间均保留。原 E2 图像在归档目录中复用，不重复生成。
- 新评价统一 `reference_v2_candidate_with_paper_fmi`；pair比较与四格比较使用不同 `comparable_group`，不能把各pair均值拼成四方法排名。
- 训练前锁定配置、划分和源码；验证检查 checkpoint、解析后训练清单及归档对照哈希。每个结果目录另存 `artifacts_sha256.json`，复用时拒绝被改写的逐图值或图像。
- 执行产生真实结果后自动登记 `EXPERIMENT_REGISTRY.json`，并调用 `maintain_experiment_index.py --check --render`。同组同时登记复用 E2 和新增候选，避免只登记候选却引用未登记对照。

本次未生成真实结果，因此没有更新注册表或重写生成表格。新增测试覆盖低频目标顺序、两源独有边、相反极性、不合法SAR支持、梯度、初始化/优化器、checkpoint推理、默认调度、产物防篡改与共同有限样本聚合；CPU检查通过，CUDA/AMP专项因当前环境没有可用CUDA跳过。效果与GPU训练稳定性仍待实际运行确认。
