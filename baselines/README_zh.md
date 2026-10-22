# 基线复现

[English](README.md) | 简体中文

[返回项目首页](../README_zh.md)

本页说明各基线的实现来源、训练数据适配、GT 用量和验证集用途，并列出论文报告运行的奖励与预算常数。本页命令均从仓库根目录执行。源码链接区分上游项目与本仓库的适配或重新实现。

## 基线复现说明

所有基线均清洗共同划分的训练行。除 DAGPipe 外，基线均不接收测试行，DAGPipe 学到的图在训练结束后也对测试输入做变换。GT 统计在训练行上读取的干净单元格数量。表中的“验证集”表示验证划分的用途。Organic 指 `flights`、`soccer` 和 `hospital`。

| 方法 | 实现与源码 | GT | 验证集 | 适配 |
|---|---|---|---|---|
| HoloClean | 适配 [HoloClean/holoclean](https://github.com/HoloClean/holoclean) | 0 | 不使用 | 使用 Clean4ML 规则文件中的 DC；Organic 数据使用 UniClean 发布的 DC；`mercedes` 每 10 列一组；`smartfactory` 按行分块；由于已发布代码会将所有字符串转为小写，`beers` 输出的字母大小写根据脏输入恢复 |
| Horizon | 适配自《Automatic Data Repair: Are We Ready to Deploy?》基准的 [Horizon 实现](https://github.com/WelkinNi/Automatic-Data-Repair/tree/main/horizon) | 0 | 不使用 | 使用 Clean4ML 规则文件中的 FD；Organic 数据使用 UniClean 发布的 FD |
| Raha+Baran | 适配 [BigDaMa/raha](https://github.com/BigDaMa/raha) | 20 条标注记录 | 不使用 | `mercedes` 分为 13 个列组，`soilmoisture` 分为 5 个列组 |
| UniClean | 清洗核心适配自 [qzkinhit/Uniclean-bench-Result](https://github.com/qzkinhit/Uniclean-bench-Result/tree/main/uniclean_cleaners) | 0 | 不使用 | 使用 Clean4ML 规则文件中的清洗器；Organic 数据使用 UniClean 发布的清洗器 |
| Lopster | [DataManagementLab/data_cleaning_with_latent_operators](https://github.com/DataManagementLab/data_cleaning_with_latent_operators)，配合本仓库适配器 | 提供 DemandClean 的 GT 预算所能覆盖的训练记录的干净副本；附录预算表中的默认档位提供一半训练行的干净副本；数据集与方法主比较表采用 DemandClean 的预算 | 不使用 | 对重建结果退化为常数的列保留输入值 |
| FM | 基于 [Narayan 等人的方法](https://github.com/HazyResearch/fm_data_tasks)进行提示词适配，使用不访问网络的 Claude Opus 4.8；[重放实现](runners/run_llm.py) | 0 | 不使用 | 读取不含标签列的脏训练行并返回单元格修改；`beers`、`breast_cancer`、`flights`、`har`、`hospitals`、`nasa` 和 `soccer` 在一次请求中读取整个训练划分；`adult`、`bike`、`mercedes`、`smartfactory` 和 `soilmoisture` 按约 25K token 分批 |
| ActiveClean | [本仓库按论文重新实现的代码](Methods/ActiveClean/)；[作者项目主页](https://activeclean.github.io/)提供的原 Python 2 版本绑定于 IMDB 文本场景 | 按整条记录提供；默认 500 条记录，上限为训练行的一半；附录预算表还包含 DemandClean 的 GT 单元格预算所允许的记录数；数据集与方法主比较表采用默认档位 | 模型选择 | 分类使用 hinge 损失的线性 SVM，回归使用平方损失的线性模型；`har` 的活动标签作为监督学习替代任务；`flights` 的时钟时间转换为分钟 |
| BoostClean | 适配 [sjyk/activedetect](https://github.com/sjyk/activedetect) | 0 | 不使用；内部留出 20% 的训练行 | `breast_cancer` 使用标签编码；`nasa`、`mercedes` 和 `soilmoisture` 的目标使用二元替代标签 1[y > median(y)] |
| SAGA | [Apache SystemDS](https://github.com/apache/systemds) 3.2.0 的 `topk_cleaning`，使用包的默认设置；评估函数适配自 [SAGA 复现脚本](https://github.com/damslab/reproducibility/tree/master/sigmod2024-SAGA) | 0 | 搜索目标 | 回归使用 `lm` 的 R²；`har` 使用 k-means 轮廓系数；`beers` 中带单位的字符串解析为数值 |
| DiffPrep | [chu-data-lab/DiffPrep](https://github.com/chu-data-lab/DiffPrep)，[提交 6499edf](https://github.com/chu-data-lab/DiffPrep/commit/6499edfbe2358288fbd0bde75f366f76da24ac27)，配合本仓库的[适配器](Methods/DiffPrep/diffprep_api.py)和[运行脚本](runners/run_diffprep/) | 0 | 搜索目标与学习率选择 | 搜索空间限定为缺失值填补和离群值清洗，输入为共同编码下保留缺失单元格的脏训练矩阵，导出训练行的软流水线输出，测试划分保持不变，回归使用标准化目标上的线性预测头并按 R² 提前停止，`mercedes`、`nasa` 和 `soilmoisture` 使用梯度裁剪，`har` 的活动标签作为替代任务 |
| Comet | [本仓库按论文重新实现的代码](Methods/Comet/)，细节参照 [sedirmohammed/COMET](https://github.com/sedirmohammed/COMET) | 由干净训练单元格生成错误掩码；默认预算为 50 个单位；附录预算表还包含与 DemandClean 初始标注记录和 GT 修复量匹配的修复；数据集与方法主比较表采用默认预算；在 `har` 上，Comet 使用从训练行中抽取的 3,000 行样本运行 | 效用分数 | 回归采用 R² 效用；`har` 使用 3 簇凝聚聚类最近质心标签的轮廓系数；`mercedes` 使用 Ridge 效用；`soilmoisture` 使用 Ridge 和 GB 效用 |
| CtxPipe | 适配 [ctxpipe/ctxpipe](https://github.com/ctxpipe/ctxpipe)，使用预训练嵌入 | 0 | 不使用 | 无 |
| ML Imp. | Clean4ML 集成的 [scikit-learn](https://github.com/scikit-learn/scikit-learn) MICE、KNN 和 RF 填补器 | 0 | 不使用 | 删除所有训练行均缺失的列，完成后再恢复这些列 |
| Mean/Med./Mode | [scikit-learn](https://github.com/scikit-learn/scikit-learn) 的 `SimpleImputer` | 0 | 不使用 | 与 ML Imp. 相同 |
| DIANA | [camillasancricca/diana-repo](https://github.com/camillasancricca/diana-repo)，提交 88e9b54，配合本仓库的[运行脚本](runners/run_diana/) | 0 | 不使用 | 按数据集下游模型所属的模型族，由知识库为每列推荐一种填补方法和一种离群值处理方法，作用于共同编码下的脏训练矩阵，回归数据集和 `har` 使用随机森林分类器的知识库，训练行中完全缺失的列用标准填补器填充 |
| Learn2Clean | [LaureBerti/Learn2Clean](https://github.com/LaureBerti/Learn2Clean) V1，提交 169bac3，配合本仓库的[补丁](Methods/Learn2Clean/learn2clean_v1.patch)和[运行脚本](runners/run_learn2clean/) | 0 | 流水线选择 | Q-learning 图限定为填补、离群值删除、去重和一致性检查，各算子只在训练行上拟合，分类、回归和 `har` 的目标模型分别为 NB、OLS 和 HCA。在 `mercedes` 上 impyute 0.0.8 的 MICE 循环始终达不到停止条件，因此唯一使用 MICE 的候选流水线以平均每个含缺失列至多十次更新的上限运行（[l2c_mice_capped.py](runners/run_learn2clean/l2c_mice_capped.py)），其他数据集均使用未修改的算子，`mercedes` 上选中的流水线不使用 MICE。 |
| RetClean | [qcri/RetClean](https://github.com/qcri/RetClean)，提交 7e72d93，使用本地 Ollama 部署的 Llama 3.1 (8B)，配合本仓库的[运行脚本](runners/run_retclean/) | 数据集与方法主比较表中以固定随机抽取的一半训练行的干净值作为数据湖，附录预算表的较低档位所用数据湖不小于全部单元格的 1.65% 与 DemandClean 的 GT 用量二者中的较大者 | 不使用 | 目标单元格来自脏训练行上的缺失标记、类型错误以及规则文件中的值域规则和 FD，按最多八个其他列语义检索最相近的三个数据湖元组，使用上游的提示词和解析器，以 `ast.literal_eval` 代替 `eval` |
| DAGPipe | 作者发布的制品，[Zenodo 10.5281/zenodo.20492518](https://doi.org/10.5281/zenodo.20492518)，配合本仓库的[运行脚本](runners/run_dagpipe/) | 0 | 训练轮次选择 | 使用作者的损失函数在脏训练行上训练由填补、编码和缩放算子构成的层次化 DAG，算子统计量只在训练行上拟合，学到的图同时变换测试输入，主比较表的模型在其输出上训练和测试，回归使用 MSE 输出头，`har` 的脏活动标签作为替代任务。GPU 训练不能逐位复现，因此发布结果来自[运行脚本](runners/run_dagpipe/README.md)在单个 CPU 线程上的运行，该运行可以逐字节复现。 |

## 各数据集的奖励与预算常数

下表列出论文报告运行的常数。RF 为包含 10 棵树、深度为 8 的随机森林；Ridge 使用 alpha = 1；KMeans 使用 3 次初始化；T 为训练轮数；`bike` 和 `nasa` 的 lambda = 1e-4。

| 数据集 | 代理模型 | T | r_max | lambda | delta_del | beta |
|---|---|---|---|---|---|---|
| `adult`, `beers` | RF | 300 | 0.3 | 0.03 | -0.02 | 0.2 |
| `breast_c.`, `smartf.` | RF | 300 | 0.3 | 0.03 | -0.02 | 0.2 |
| `bike` | RF | 200 | 1.0 | 1e-4 | -0.05 | 1.0 |
| `mercedes` | Ridge | 300 | 0.3 | 0.03 | -0.05 | 1.0 |
| `nasa` | RF | 200 | 1.0 | 1e-4 | -0.05 | 1.0 |
| `soilmoist.` | Ridge | 300 | B = 0 | 0.03 | -0.05 | 1.0 |
| `har` | KMeans | 300 | 0.3 | 0.1 | -0.02 | 0.2 |
| `flights` | RF | 300 | 0.3 | 0.03 | -0.02 | 0.2 |
| `soccer`, `hospital` | RF | 300 | B = 0 | 0.03 | -0.02 | 0.2 |

## FM 复现与费用

FM 使用不访问网络的 Claude Opus 4.8，仅读取去掉标签列后的脏训练行。按与其他方法相同的 GT 统计规则，其 GT 用量为 0。

报告的 API 费用计入每个最终采用运行的全部轮次。每轮重复读取的上下文按缓存读取价计费，创建缓存按相应写入价计费。Opus 4.8 每百万 token 的价格如下。

| Token 类别 | 每百万 token 费用（美元） |
| --- | ---: |
| 未缓存的输入 | $5.00 |
| 缓存写入，有效期 5 分钟 | $6.25 |
| 缓存写入，有效期 1 小时 | $10.00 |
| 缓存读取 | $0.50 |
| 输出 | $25.00 |

十二个数据集报告的总费用为 **$207.30**，单个数据集费用最高的是 **mercedes，为 $98.27**。各数据集的用量和费用记录于 [results/llm/llm_cost.json](../results/llm/llm_cost.json)。

LLM 基线可以离线重放所附响应。

```sh
python baselines/runners/run_llm.py --dataset mercedes --check
```

该命令核对重放结果与发布输出是否一致，不调用 API。采用输出所对应的请求、响应和计费记录位于 [results/llm/](../results/llm/)。

## 基线与许可证

HoloClean 的最终 `beers` 输出也可以通过 [restore_case.py](runners/run_holoclean/restore_case.py)，按照已发布的[运行清单](../results/main_table/beers/baselines/holoclean/manifest.json)重放。该固定后处理步骤使用完整脏输入的词汇表恢复大小写，不需要干净值。

[runners/run_lopster.py](runners/run_lopster.py) 提供 Lopster 的固定输入准备流程和最终输出转换。`--prepare-only` 和 `--postprocess-only` 均不需要上游代码或模型训练。[Lopster 配置说明](Methods/Lopster/LICENSE_NOTE.md) 列出了输入格式、固定设置和源代码配置方法。

各基线的专用依赖、适配器和许可证列于 [THIRD_PARTY.md](THIRD_PARTY.md)。SAGA 需要 Apache SystemDS 3.2.0 和 Java 17。HoloClean 需要 PostgreSQL，CtxPipe 需要 GTE-large 嵌入模型。Lopster 和 DiffPrep 的上游代码未发布允许再分发的许可证，因此需要单独获取；各自的许可证说明指明了所需源码。DIANA、Learn2Clean、RetClean 和 DAGPipe 的上游代码同样需要按上表所列版本单独获取，其运行脚本读取的目录见 [THIRD_PARTY.md](THIRD_PARTY.md)。Learn2Clean 需要应用本仓库的补丁后运行。运行清单和适配器文档记录了本仓库采用的配置。

本仓库自行编写的代码适用 [LICENSE](../LICENSE)。数据集和第三方代码保留各自的许可证，详情见 [data/README.md](../data/README.md) 和 [THIRD_PARTY.md](THIRD_PARTY.md)。
