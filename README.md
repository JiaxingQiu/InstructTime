# Instruction-based Time Series Editing

A short description of your project and what it does.

## Table of Contents

- [Overview](#overview)
- [Contribution](#contribution)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

[Brief overview of the project. Explain the problem you're solving, and why this project matters.]

## Contribution

### 1
### 2
### 3

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/JiaxingQiu/InstructTime.git
cd InstructTime
pip install -r requirements.txt
```
## Datasets

Download data.zip from [Hugging Face](https://huggingface.co/datasets/JiaxingJoy/InstructTime/tree/main).
```bash
# or in python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="JiaxingJoy/InstructTime", filename="data.zip", repo_type="dataset")
```
Download results.zip from [Hugging Face](https://huggingface.co/datasets/JiaxingJoy/InstructTime/tree/main).


## Usage

Run the following command for each dataset.  
Use optional flags to control the behavior:
- `--open_vocab` for open-vocabulary editing
- `--overwrite` to retrain the model
- `--attr_suffix _at` to enable attribute-based editing

```bash
# evaluate model from checkpoint
python main.py --dataset_name air # air quality
python main.py --dataset_name syn_gt # synthetic with ground truth
python main.py --dataset_name syn # synthetic
python main.py --dataset_name nicu # NICU heart rate
```



Or for Jupyter notebooks:

```bash
jupyter notebook
```

## Examples

Provide minimal code examples or screenshots demonstrating how to use the project.

## Project Structure

```
your-repo/
├── src/                  # Source code
├── notebooks/            # Jupyter notebooks
├── data/                 # Input data or datasets
├── tests/                # Unit tests
├── README.md
├── requirements.txt
└── LICENSE
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## License

[MIT](LICENSE) © Jiaxing Qiu
