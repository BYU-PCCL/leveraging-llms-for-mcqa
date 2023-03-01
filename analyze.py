from argparse import ArgumentParser
from experiment_config import ExperimentConfig
from scipy import stats
from utils import idx_to_ltr

import numpy as np
import pandas as pd


def get_config_from_args():
    parser = ArgumentParser()
    parser.add_argument("ds_name", help="Dataset name")
    parser.add_argument("model_name", help="Model name")
    parser.add_argument("style_name", help="Style name")
    parser.add_argument("n_shots", type=int, help="# of shots")
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
    return ExperimentConfig(**vars(args))


def add_correct_answer_col(df):
    df["correct_answer"] = df.apply(
        lambda row: idx_to_ltr(row["qwe"]["answer_idx"]),
        axis=1
    )


def div_dicts(a, b):
    # Divide each value in dictionary a by the matching
    # value in dictionary b
    new_dict = dict()
    for key in a.keys():
        if key in b.keys():
            new_dict[key] = a[key] / b[key]
    return new_dict


def sub_dicts(a, b):
    # Subtract from each value in dictionary a the matching
    # value in dictionary b
    new_dict = dict()
    for key in a.keys():
        if key in b.keys():
            new_dict[key] = a[key] - b[key]
    return new_dict


def analyze_results(config):
    # Get file name of experiment to load
    fname = config.get_save_fname()

    # Load file
    df = pd.read_pickle(fname)

    if config.style_name == "natural":
        if config.do_perm:
            # We start by calculating the logprob of each
            # answer option irrespective of the order the
            # options were presented in
            def get_lp(ltr, lps):
                if f"Ġ{ltr}" in lps.keys():
                    return lps[f"Ġ{ltr}"]
                elif f" {ltr}" in lps.keys():
                    return lps[f" {ltr}"]
                else:
                    return -np.inf

            df["ord_lps"] = df.apply(
                lambda row: [
                    get_lp(
                        idx_to_ltr(row['perm_order'].index(i)).upper(),
                        row["model_response"]["logprobs"]
                    ) for i in range(len(row["perm_order"]))],
                axis=1
            )

            df["coverage"] = df.apply(
                lambda row: np.sum(np.exp(row["ord_lps"])),
                axis=1
            )
            print(f"Coverage: {df['coverage'].mean()}")

            # Add a column for if model got question right
            df["correct"] = df.apply(
                lambda row: max(
                    row["model_response"]["logprobs"].items(),
                    key=lambda x: x[1]
                    # In line below [0] is the key (as opposed to value)
                    # Additionally we use 1: instead of lstrip because
                    # we want the prediction "A" to be wrong when " A"
                    # is expected, for example
                )[0][1:] == idx_to_ltr(row["qwe"]["answer_idx"]),
                axis=1
            )
            print(f"Accuracy: {df['correct'].mean()}")

            # Making lists of lists
            grouped = df.groupby("question_idx")["ord_lps"].apply(list)
            lps_by_question = grouped.tolist()

            # HOW MANY OF THE CHOSEN ANSWERS MATCH THE MAJORITY
            # ANSWER?
            props = list()
            for q_lps in lps_by_question:
                majority_choice = stats.mode(
                    [np.argmax(x) for x in q_lps]
                )[0][0]
                props.append(
                    sum(
                        [np.argmax(x) == majority_choice for x in q_lps]
                    ) / len(q_lps)
                )

            print("PPA:", np.mean(props))

        else:
            add_correct_answer_col(df)
            df["chosen_answer_raw"] = df.apply(
                lambda row: max(
                    row["model_response"]["logprobs"].items(),
                    key=lambda x: x[1]
                    # In line below [0] is the key (as opposed to value)
                    # Additionally we use 1: instead of lstrip because
                    # we want the prediction "A" to be wrong when " A"
                    # is expected, for example
                )[0][1:],
                axis=1
            )

            df["correct"] = df.apply(
                lambda row: row["chosen_answer_raw"] == row["correct_answer"],
                axis=1
            )

            print(
                "Accuracy:",
                df["correct"].mean()
            )

            # If config.ds_name == "mmlu" we'll present accuracy
            # after grouping by "task"
            if config.ds_name == "mmlu":
                print("Accuracy by task:")
                g = df.groupby("task")["correct"].mean()
                for i, task_name in enumerate(g.index):
                    print(task_name, round(g[i]*100, 1))
    else:
        add_correct_answer_col(df)
        df["chosen_answer_raw"] = df.apply(
            lambda row: max(
                row["model_response"]["logprobs"].items(),
                key=lambda x: x[1]
                # In line below [0] is the key (as opposed to value)
                # No need for 1: here because we assign the letters
                # manually in models.py
            )[0],
            axis=1
        )
        print(
            "Accuracy (raw):",
            (df["chosen_answer_raw"] == df["correct_answer"]).mean()
        )

        # Answer with length normalization
        df["chosen_answer_ln"] = df.apply(
            lambda row: max(
                div_dicts(
                    row["model_response"]["logprobs"],
                    row["model_response"]["lens"]
                ).items(),
                key=lambda x: x[1]
            )[0],
            axis=1
        )
        print(
            "Accuracy (length-normalized):",
            (df["chosen_answer_ln"] == df["correct_answer"]).mean()
        )

        # Answer with special normalization
        df["chosen_answer_sn"] = df.apply(
            lambda row: max(
                sub_dicts(
                    row["model_response"]["logprobs"],
                    row["model_response"]["unconditional_logprobs"]
                ).items(),
                key=lambda x: x[1]
            )[0],
            axis=1
        )
        print(
            "Accuracy (unconditional-normalized):",
            (df["chosen_answer_sn"] == df["correct_answer"]).mean()
        )


if __name__ == "__main__":
    analyze_results(get_config_from_args())
