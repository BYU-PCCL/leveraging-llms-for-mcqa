from argparse import ArgumentParser
from dataset_utils import get_dataset_info, get_questions_with_exemplars

import random


def get_config_from_args():
    parser = ArgumentParser()
    parser.add_argument("ds_name", help="Dataset name")
    parser.add_argument("style_name", help="Style name")
    parser.add_argument("n_shots", type=int, help="# of shots")
    parser.add_argument("--longest", action="store_true")
    args = parser.parse_args()
    return vars(args)


def viz_prompts(ds_name, style_name, n_shots, longest):

    # Get questions with exemplars
    qwes = get_questions_with_exemplars(
        info=get_dataset_info(ds_name),
        n_shots=n_shots,
        do_strong_shuffle=False
    )

    if style_name == "natural":
        prompt_texts = [q.get_natural_prompt() for q in qwes]
    elif style_name == "brown":
        prompt_texts = [q.get_brown_prompt() for q in qwes]

    if longest:
        p = max(prompt_texts, key=len)
    else:
        random.seed()
        p = random.choice(prompt_texts)
    print(p)


if __name__ == "__main__":
    viz_prompts(**get_config_from_args())
