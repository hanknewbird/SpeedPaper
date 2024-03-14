# SpeedPaper

## 项目简介

SpeedPaper 是一个旨在帮助深度学习初学者和爱好者更容易理解和掌握复杂研究论文的项目。我们通过提供原论文的中文翻译和相应的PyTorch代码复现，使得读者能够快速入门并深入理解每篇论文的核心概念和技术细节。

## 为什么选择SpeedPaper？

- **易读性**：每篇论文的代码都是独立构建的(数据集除外)，确保了阅读和理解的连贯性。
- **详尽注释**：代码中几乎每段都附有注释，帮助读者理解每一行代码的目的和作用。
- **易于理解**：在容易混淆的概念或实现上，我们特别添加了注释和解释，以降低理解难度。

## 如何使用SpeedPaper？

1. **阅读论文翻译**：在每篇论文的`paper`目录下找到您感兴趣的论文翻译，开始您的学习之旅。
2. **查看代码实现**：在每篇论文的根目录下，您可以找到对应论文的PyTorch代码实现。
3. **运行示例**：请参考每个代码目录下的`README.md`文件，了解如何运行和测试代码。
4. **参与贡献**：如果您发现翻译或代码中有待改进之处，欢迎提交Pull Request。

# BaseLne

| 论文提出时间 | 论文名称                              | 历史地位或特点                | 已完成 |
|--------|-----------------------------------|------------------------|-----|
| 2012年  | AlexNet                           | 深度学习热潮的奠基作             | ✅   |
| 2014年  | VGG                               | 使用3x3卷积构造更深的网络         | ❌   |
| 2014年  | GoogLeNet v1 (Inception v1)       | 使用并行架构构造更宽的网络          | ❌   |
| 2015年  | GoogLeNet v2 (BatchNormalization) | 规范化层的输入，改善训练过程         | ❌   |
| 2015年  | GoogLeNet v3 (Inception v3)       | 进一步优化的Inception模块，提升性能 | ❌   |
| 2015年  | ResNet                            | 构建深层网络都要有的残差连接         | ❌   |
| 2016年  | GoogLeNet v4                      | 继续优化的GoogLeNet版本       | ❌   |
| 2016年  | ResNeXt                           | 分组卷积，提高性能和扩展性          | ❌   |
| 2017年  | DenseNet                          | 特征重用，显著提高效率和性能         | ❌   |
| 2017年  | SENet                             | 通道间依赖关系的建模，提升准确性和鲁棒性   | ❌   |

# Segmentation

| 论文提出时间 | 论文名称                                            | 历史地位或特点           | 已完成 |
|--------|-------------------------------------------------|-------------------|-----|
| 2012年  | FCN (Fully Convolutional Networks)              | 全卷积网络，用于图像分割任务    | ❌   |
| 2015年  | Unet                                            | 用于医学图像分割的U-Net变体  | ❌   |
| 2015年  | SegNet                                          | 基于SegNet的深度卷积生成网络 | ❌   |
| 2016年  | DeepLab                                         | 深度学习在图像分割领域的里程碑   | ❌   |
| 2017年  | GCN (Graph Convolutional Networks)              | 图卷积网络的开创性工作       | ❌   |
| 2018年  | DFN (Deep Feature Network)                      | 用于图像去噪的深度特征网络     | ❌   |
| 2018年  | ENet (Efficient Network)                        | 高效的网络结构，用于图像分割    | ❌   |
| 2019年  | BiSeNet (Bilateral Segmentation Network)        | 双边网络结构，用于图像分割     | ❌   |
| 2019年  | DFANET (Dual Fusion Aggregation Network)        | 用于图像分割的双融合聚合网络    | ❌   |
| 2020年  | RedNet (Rethinking Dilated Convolution)         | 重新思考空洞卷积的网络结构     | ❌   |
| 2020年  | RDFNet (Relation-Aware Dynamic Feature Network) | 关系感知动态特征网络，用于图像分割 | ❌   |

