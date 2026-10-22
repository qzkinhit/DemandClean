# DemandClean

[English](README.md) | 简体中文

DemandClean 学习面向下游模型的数据清洗策略。对每个检测到的错误，策略可以保留原值、请求真值修复、删除整行，或用估计值替换。

本仓库提供论文的实现、最终配置、数据集、输出结果及复现记录。比较实验覆盖十二个数据集、十八个基线和四种参照策略。补充预算比较在两个 GT 档位下共 78 次运行，ActiveClean、Comet 和 Lopster 覆盖九个数据集，RetClean 覆盖全部十二个数据集。

## 快速开始

请使用 Python 3.8 或更新版本，并创建独立环境。统一下游评分器的参考版本为 scikit-learn 1.3.2。

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scikit-learn==1.3.2
```

检查发布记录并重放所附 LLM 结果，无需训练模型或调用 API：

```sh
python evaluation/collect_results.py --check
python baselines/runners/run_llm.py --dataset breast_cancer --check
```

使用最终 breast_cancer 配置训练并清洗：

```sh
python scripts/run_demandclean.py --config configs/datasets/breast_cancer.yaml \
    --variant full --out results/runs/breast_cancer
```

复现时请采用各运行 `manifest.json` 中记录的参数和随机种子。使用已有模型进行推理、下游评分、归因分析和图表生成的命令见[复现指南](docs/CODE_TOUR.md)。

## 详细文档

- [基线复现](baselines/README_zh.md)：源码仓库、实现适配、GT 与验证集用途、奖励与预算常数，以及 FM 费用。
- [发布结果](results/README.md)：最终输出、运行记录及[结果索引](results/index.json)。
- [代码与评估指南](docs/CODE_TOUR.md)：实现结构、评估口径，以及论文表格和插图的生成命令。
- [FM 重放与日志](results/llm/README.md)：提示词、响应与 token 统计。
- [图表目录](figures/)：绘图脚本与最终输入。
- 论文附录：完整证明、质量状态的充分性分析、方法细节、实现细节与补充结果（[English](docs/appendix.pdf)、[中文](docs/appendix_zh.pdf)）。

核心实现位于 [demandclean/](demandclean/)，数据集配置位于 [configs/](configs/)，数据和规则库位于 [data/](data/)。

## 许可证

本仓库自行编写的代码适用 [LICENSE](LICENSE)。第三方代码和数据集保留各自的许可证，详情见[基线许可证说明](baselines/THIRD_PARTY.md)和[数据集文档](data/README.md)。
