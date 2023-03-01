# leveraging-llms-for-mcqa

## Overview
This is the code for the ICLR 2023 paper "[Leveraging Large Language Models for Multiple Choice Question Answering](https://arxiv.org/abs/2210.12353)." It can be used to reproduce results in the paper and is designed to be extensible.

## Setup
* Start by using your favorite package manager to install `datasets`, `numpy`, `openai`, `pandas`, `scipy`, `tqdm`, and `transformers`.
* Now register your API keys in `api_sectrets.py`. To do this, add a key and value for each API key you want to register to the dictionary in the `get_api_key_by_name` function. You'll need an OpenAI key for OpenAI API experiments, and a Jurassic key for Jurassic API experiments. You can use the existing keys or choose your own names for the keys.

## Running Experiments
To run experiments and reproduce the results from the paper you will use `main.py`.

The positional command line arguments are:
* The name of the dataset to use (must be a key from the dictionary inside `get_dataset_info` in `dataset_utils.py`) e.g., "mmlu"
* The name of the model to use (must be a key in one of the dictionaries in `get_model_by_name` in `models.py`) e.g., "codex"
* The name of the prompting style to use (either "brown" (called CP in the paper) or "natural" (called MCP in the paper)
* The number of shots to use ("0" for zero-shot, "1" for one-shot, etc.)
* The name of the API key to use (must be a key from the dictionary inside `get_api_key_by_name` in `api_secrets.py`

The optional command line arguments are:
* `--do_strong_shuffle`: For strong shuffling as used in Appendix C
* `--do_perm`: For passing all permutations of each question to the model, as in the experiments in Section 4

Running `main.py` will save a pickle file with experiment results.

## Analyzing Results
To analyze the results of an experiment (from its saved pickle file) you will use `analyze.py`. The positional and optional command line arguments are the same except for you don't need to supply the name of an API key to use. These arguments will be used to look up the saved experiment pickle file.

## Other Functionality
* You can **visualize prompts** that will be used by an experiment with `viz_prompts.py`. The positional command line arguments are dataset name, style name, and number of shots (as you'd use with `main.py`). The optional argument `--longest` will show the longest prompt instead of a random one.
* You can **add a custom model** by adding a custom key and value to a dictionary in `get_model_by_name` within `models.py`.
* You can **add a custom dataset** by adding a custom key and value to the dictionary in `get_dataset_info` within `dataset_utils.py`.
