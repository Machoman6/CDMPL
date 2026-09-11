# CDMPL

**Chinese Cyberbullying Detection via Multi-Feature Prompt Learning**

面向中文网络暴力文本检测的少样本提示学习代码。项目基于 [OpenPrompt](https://github.com/thunlp/OpenPrompt)，使用中文预训练语言模型、模板（template）和标签词（verbalizer）完成二分类实验，并提供基于知识扩展的 `KPT` verbalizer 实现。

> 说明：本仓库包含数据、训练脚本和提示组件，但不包含预训练模型权重、独立实验结果表或论文全文。README 中未填入未经仓库验证的性能指标。

## Overview

### Task

给定一条中文事件描述，判断其是否属于网络暴力事件。当前数据集包含两个类别：

| Label | Meaning |
| ---: | --- |
| `0` | 非网暴事件 |
| `1` | 网暴事件 |

### Method

代码的主要实验流程如下：

1. 读取 `datasets/TextClassification/cyberbullying/` 下的训练集和测试集。
2. 使用 `FewShotSampler` 按类别抽取少量训练样本，并根据随机种子划分验证集。
3. 使用 `scripts/TextClassification/cyberbullying/` 中的模板和 verbalizer 构造提示学习任务。
4. 通过 OpenPrompt 的 `PromptForClassification` 微调预训练语言模型。
5. 在测试集上计算 Accuracy、Macro-Precision、Macro-Recall 和 Macro-F1，并将结果追加写入指定文件。

## Repository Structure

```text
.
├── datasets/TextClassification/cyberbullying/
│   ├── all_data.csv       # 完整数据
│   ├── train.csv          # 训练数据
│   ├── test.csv           # 测试数据
│   └── class.txt          # 类别名称
├── models/enhanced_model.py
├── prompts/knowledgeable_verbalizer.py
├── scripts/TextClassification/cyberbullying/
│   ├── manual_template.txt
│   ├── ptuning_template.txt
│   ├── manual_verbalizer.txt
│   └── kpt_translate_verbalizer.txt
├── fewshot.py             # 单次实验入口
├── auto_run.py            # 批量实验入口
├── requirements.txt
└── autoformer-reproduct-pipeline_base.txt
```

## Requirements

- Linux or WSL is recommended.
- Python 3.9.x; the original environment used Python 3.9.20.
- CUDA-capable GPU is required by the current implementation because `fewshot.py` unconditionally enables CUDA.
- A local Chinese pretrained language model checkpoint. The default path is `../Model/bert_base_chinese`.

The dependency file contains a pinned research environment and includes OpenPrompt, PyTorch, Transformers and related packages. Some entries are Conda-build paths, so installation may need adjustment for another operating system or Python environment.

## Installation

```bash
conda create -n cdmpl python=3.9.20
conda activate cdmpl
pip install -r requirements.txt
```

Prepare a compatible Chinese BERT checkpoint, then keep it at the default path or pass its path through `--model_name_or_path`.

## OpenPrompt Compatibility Patch

The repository includes a customized `pipeline_base.py` implementation as `autoformer-reproduct-pipeline_base.txt`. Copy it into the active environment's OpenPrompt package directory:

```bash
OPENPROMPT_DIR=$(python -c "import openprompt, os; print(os.path.dirname(openprompt.__file__))")
cp autoformer-reproduct-pipeline_base.txt "$OPENPROMPT_DIR/pipeline_base.py"
```

## Data Format

The CSV files are consumed by OpenPrompt's `CbProcessor`. The repository's examples use the following fields:

```text
title,description,label,sequence
```

`label` is an integer class label, while `sequence` stores the token or feature sequence used by the dataset. Keep the existing header and encoding when replacing data. Do not commit sensitive or personally identifiable information to a public repository.

## Running Experiments

Run one few-shot experiment from the repository root:

```bash
python fewshot.py \
  --dataset cyberbullying \
  --model bert \
  --model_name_or_path ../Model/bert_base_chinese \
  --manual_or_soft manual \
  --template_id 1 \
  --verbalizer kpt \
  --shot 15 \
  --seed 143 \
  --batch_size 16 \
  --learning_rate 4e-5 \
  --max_epochs 20 \
  --result_file ./results.txt
```

The configured batch runner uses the same default experiment setting:

```bash
python auto_run.py
```

Implemented verbalizer options include `kpt`, `manual`, `trpt`, and `kpt_plus`, subject to the corresponding files under `scripts/TextClassification/cyberbullying/`. The current `auto_run.py` uses `kpt`.

## Important Arguments

| Argument | Description | Current default |
| --- | --- | --- |
| `--dataset` | Dataset name | unset; use `cyberbullying` |
| `--model` | OpenPrompt model family | `bert` |
| `--model_name_or_path` | Local pretrained model path | `../Model/bert_base_chinese` |
| `--template_id` | Template index | unset; use `1` |
| `--verbalizer` | Label-word strategy | unset; use `kpt` or `manual` |
| `--shot` | Examples per class | `5` |
| `--seed` | Random seed | `144` |
| `--batch_size` | Batch size | `16` |
| `--max_epochs` | Maximum training epochs | `20` |
| `--result_file` | Result output file | script-defined path |

## Outputs and Evaluation

The scripts may create or append to `results.txt` (or `--result_file`) and create `out_label.csv` containing gold labels and predictions. For fair comparison, report the checkpoint, data split, shots, template, verbalizer, seed, learning rate, batch size, epochs, GPU and metrics. For few-shot experiments, repeat multiple seeds and report mean and standard deviation where possible.

## Reproducibility Checklist

- Use the same Python, PyTorch, Transformers, OpenPrompt versions and pretrained checkpoint.
- Run commands from the repository root so relative paths resolve correctly.
- Keep split, `--shot`, template, verbalizer and seed fixed when comparing methods.
- Record GPU type, CUDA version and all experiment arguments.
- Do not treat a single random seed as a statistically robust estimate.

## Limitations

- Only the `cyberbullying` dataset is currently wired into `fewshot.py`.
- The implementation assumes CUDA and does not expose a CPU fallback.
- The pretrained model is referenced by a local path and is not distributed here.
- No formal configuration file, automated test suite, or checked-in benchmark table is provided.
- The dataset may contain sensitive user-generated content; follow applicable data and privacy requirements.

## Citation

Replace the placeholder below with verified bibliographic information before publication:

```bibtex
@article{cdmpl,
  title   = {Chinese Cyberbullying Detection via Multi-Feature Prompt Learning},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

This project builds on [OpenPrompt](https://github.com/thunlp/OpenPrompt). Please also cite OpenPrompt, the pretrained model, and the dataset according to their official instructions.

## License

No license file is currently included. Add an explicit license before allowing reuse or redistribution.
