# InstructTime

- [Overview](#overview)
- [Contribution](#contribution)
- [Installation](#installation)
- [Use InstructTime](#use-instructtime)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

## Overview

We introduced Instruction-based Time Series Editing, a novel task that enables fine-grained modification of time series based on natural language instructions. Unlike prior attribute-based approaches, our formulation allows flexible conditioning on free-form text, aligning more closely with real-world scenarios where text captures nuanced, individualized context.

## Contributions

- We introduce instruction-based time series editing, enabling users to modify time series using natural language instructions.
- We propose *InstructTime*, the first model for this task, supporting high-fidelity, multi-resolution editing with controllable strength, which accounts for the discrete nature of language.
- InstructTime generalizes to unseen expressions in a zero-shot setting, and adapts to unseen conditions with few-shot tuning.

## Installation

Download this repository and install dependencies:

```bash
cd InstructTime
pip install -r requirements.txt
```

## Datasets

Download data.zip from [Hugging Face](https://huggingface.co/datasets/JJoy333/InstructTime), extract it, and place the `data/` folder under the `InstructTime` directory.

Download results.zip from [Hugging Face](https://huggingface.co/datasets/JJoy333/InstructTime), extract it, and place it under `InstructTime/script/VITAL/` for InstructTime model checkpoints.

## Use InstructTime

Backbone architecture of InstructTime is provided in `instructtime_backbone/`.

## Experiments

To re-run experiments, after placing the `data/` folder, run the following command for each dataset, with optional flags to control modeling type:

- `--open_vocab` for open-vocabulary (unseen expressions) editing
- `--overwrite` to retrain the model
- `--attr_suffix _at_level` to input condition as attributes
- `--gamma 10` hyperparameter ratio of unit decrease in contrustive loss over unit decrease in reconstruction loss, can be set to 1, 10, 100
- `--suffix _your_special_suffix` customized model name suffix, default empty string

Results will be saved under ./script/VITAL/results/{dataset_name}{attr_suffix}{suffix}/

```bash
cd ./script/VITAL

# evaluate model from checkpoint
python main.py --dataset_name air # air quality
python main.py --dataset_name syn_gt # synthetic with ground truth
python main.py --dataset_name nicu # NICU heart rate

# train instruction-based model from scratch
python main.py --dataset_name air --overwrite
python main.py --dataset_name syn_gt --overwrite
python main.py --dataset_name nicu --overwrite

# train attribute-based version
python main.py --dataset_name air --overwrite --attr_suffix _at_level
python main.py --dataset_name syn_gt --overwrite --attr_suffix _at_level
python main.py --dataset_name nicu --overwrite --attr_suffix _at_level

```

Or run jupyter notebook: notebooks/main_instructtime.ipynb


## Citation

If you find this useful, please cite:

>Jiaxing Qiu, Dongliang Guo, Brynne Sullivan, Teague R. Henry, and Thomas Hartvigsen. 2026. *Instruction-based Time Series Editing.* KDD ’26. https://doi.org/10.1145/3770854.3780299

<details>
<summary>BibTeX</summary>
```bibtex
@inproceedings{qiu2026instructtime,
  title     = {Instruction-based Time Series Editing},
  author    = {Qiu, Jiaxing and Guo, Dongliang and Sullivan, Brynne and Henry, Teague R. and Hartvigsen, Thomas},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD '26)},
  year      = {2026},
  doi       = {10.1145/3770854.3780299},
  url       = {https://doi.org/10.1145/3770854.3780299}
}
```
</details>

## License

This project is licensed under the MIT License. See `LICENSE`.
