from api_secrets import get_api_key_by_name
from argparse import ArgumentParser
from constants import SAVE_EVERY
from dataset_utils import get_dataset_info, get_questions_with_exemplars
from experiment_config import ExperimentConfig
from experiment_saver import ExperimentSaver
from itertools import permutations
from models import get_model_by_name
from tqdm import tqdm

import copy


def get_config_and_api_key_name_from_args():
    parser = ArgumentParser()
    parser.add_argument("ds_name", help="Dataset name")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("style_name", help="Style name")
    parser.add_argument("n_shots", type=int, help="# of shots")
    parser.add_argument("api_key_name", help="API key name")
    parser.add_argument(
        "--do_strong_shuffle",
        action="store_true",
        help="Force correct answer index to change for each example"
    )
    parser.add_argument(
        "--do_perm",
        action="store_true",
        help="Process every example with all possible answer orderings"
    )
    args = parser.parse_args()
    api_key_name = args.api_key_name
    args = vars(args)
    del args["api_key_name"]
    return ExperimentConfig(**args), api_key_name


def run_experiment(config, api_key_name):

    # Get API key
    api_key = get_api_key_by_name(name=api_key_name)

    # Load model
    model = get_model_by_name(
        name=config.model_name,
        api_key=api_key
    )
    model = {
        "natural": model.process_question_natural,
        "brown": model.process_question_brown
    }[config.style_name]

    # Get questions with exemplars
    qwes = get_questions_with_exemplars(
        info=get_dataset_info(config.ds_name),
        n_shots=config.n_shots,
        do_strong_shuffle=config.do_strong_shuffle
    )

    # Run experiment, saving results
    saver = ExperimentSaver(save_fname=config.get_save_fname())
    for q_idx, qwe in enumerate(tqdm(qwes)):

        if config.do_perm:
            for perm_order in permutations(range(qwe.get_n_choices())):
                qwe_copy = copy.deepcopy(qwe)
                qwe_copy.permute_choices(perm_order)
                response = model(qwe_copy)
                saver["question_idx"].append(q_idx)
                saver["perm_order"].append(perm_order)
                saver["qwe"].append(vars(qwe_copy))
                saver["model_response"].append(vars(response))

            # When doing permutations we ignore SAVE_EVERY and
            # save after every question
            saver.save()
        else:
            response = model(qwe)

            saver["question_idx"].append(q_idx)
            if qwe.task is not None:
                saver["task"].append(qwe.task)
            saver["qwe"].append(vars(qwe))
            saver["model_response"].append(vars(response))

            if q_idx % SAVE_EVERY == 0 and q_idx != 0:
                saver.save()

    saver.save()


if __name__ == "__main__":
    run_experiment(*get_config_and_api_key_name_from_args())
