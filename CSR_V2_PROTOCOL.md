# CSR-v2：校准SPD证据与Logit竞争路由

## 修改依据

CSR-v1 validation1000 表明光学缺失图定位正确，但 SPD 删除图近乎全黑、SAR可靠图空间变化不足，三重乘法门控将 Optical Gate 压至 0.584%，输出修正 RMS 仅为 0.000114。CSR-v2只修改证据压缩和门控，不改变10%中频、3%高频替换上限。

## 相对CSR-v1的变化

1. SPD删除证据改为逐通道相对删除率，使用Top-12.5%通道聚合。
2. 使用图像内50%/95%百分位校准，并通过最大池化保留稀疏线结构。
3. SAR可靠性同时考虑清洗特征显著性、保留比例和Raw/Clean方向一致性。
4. Optical缺失、SPD删除和SAR可靠性作为logit加减项进入Softmax，不再与Optical概率连续相乘。
5. Optical Gate全图均值目标设为4%，并记录大于0.1的覆盖率。
6. 新增缺失区域门控召回率、可靠SAR区域误激活率、SPD证据空间标准差和输出修正RMS。

## 不变部分

- B0、Detail-only SPD、Encoder、融合层、Decoder全部冻结。
- 初始输出严格等于Detail-only SPD。
- 中频最大替换0.10，高频最大替换0.03。
- 数据划分、检查点和14项论文指标不变。

## Validation1000判断标准

- NAB/F相对下降至少5%。
- FMI下降不超过0.2%。
- QAB/F下降不超过0.1%。
- VIF下降不超过0.2%。
- Optical Gate均值应接近2%至6%。
- SPD删除图和SAR可靠图必须具有明确空间变化，不能接近全黑或均匀灰。
- 可靠SAR区域误激活应显著低于光学缺失区域门控召回。
- 四联图必须出现局部可辨的光学道路、田界或建筑轮廓恢复。

不满足时不得进入正式test1000。

## 命令

- 训练：`run_csr_v2_training_seed42.cmd`
- 验证：`run_csr_v2_validation.cmd`
- 正式测试：仅验证通过后执行 `run_csr_v2_test1000.cmd`
