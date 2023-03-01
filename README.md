# leveraging-llms-for-mcqa

## Setup
* Start by using your favorite package manager to install: `openai`, `pandas`, `tqdm`,
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
