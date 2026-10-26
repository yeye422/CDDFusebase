# PDCF P0：C0 兼容骨架与恒等性验证

## 目的

P0 只建立 PDCF 的模型接口，不进行训练。新增网络包括：

- 可逆 Base/Detail 重分配；
- 双向跨模态预测器；
- Correspondence Fusion；
- Complementary Fusion；
- 输出为 `delta_base`、`delta_detail` 的联合投影。

联合投影最后一层使用全零初始化，因此初始状态满足：

```text
PDCF output == C0 output
```

C0 Encoder、Decoder、BaseFuse 和 DetailFuse 默认全部冻结。

## 单元检查

当前虚拟环境未安装 pytest，可直接执行测试文件中的测试函数：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe -c "import tests.test_pdcf_p0 as t; fs=[getattr(t,n) for n in dir(t) if n.startswith('test_')]; [f() for f in fs]; print(f'{len(fs)} PDCF P0 tests passed')"
```

检查内容：

- Refactor 零初始化恒等性；
- Refactor 正向/逆向恢复；
- Base/Detail 归一化统计接口；
- PDCF 初始化输出与 C0 完全一致；
- C0 参数冻结；
- 初始梯度能够到达联合输出层；
- SPD 条件特征的通道和空间尺寸接口。

## 真实 C0 单样本验证

该命令只处理一个随机小张量，不属于训练或批量测试：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe verify_pdcf_c0_identity.py
```

如需保存 P0 元数据：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe verify_pdcf_c0_identity.py --output tmp\pdcf_p0_identity\summary.json
```

脚本保存或打印：checkpoint 路径及 SHA-256、输入尺寸、最大输出误差、Base/Detail 修正量、冻结状态和参数量。

## P0 边界

P0 不证明可预测分解有效，也不证明双路径融合优于普通卷积。下一阶段 P1 才训练容量受限的双向预测器并进行预测残余干预实验。

