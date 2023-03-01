from constants import HF_CACHE_DIR_NAME, MMLU_NAMES, REPRODUCIBILITY_SEED
from dataclasses import dataclass
from datasets import load_dataset
from itertools import groupby
from prompts import Exemplar, QuestionPart, QuestionWithExemplars
from typing import Callable
from utils import idx_to_ltr, ltr_to_idx

import random


@dataclass
class DatasetInfo:
    path: str
    exemplar_split: str
    eval_split: str
    extractor: Callable
    name: str = None
    data_dir: str = None


def load_hf_dataset(path, name, data_dir, split):
    if split.endswith(".jsonl"):
        # This is for Social IQa test sets
        return load_dataset("json", data_files=split)["train"]
    else:
        return load_dataset(
            path=path,
            name=name,
            data_dir=data_dir,
            split=split,
            cache_dir=HF_CACHE_DIR_NAME
        )


def load_hf_dataset_no_verify(path, name, data_dir, split):
    return load_dataset(
        path=path,
        name=name,
        data_dir=data_dir,
        split=split,
        cache_dir=HF_CACHE_DIR_NAME,
        ignore_verifications=True
    )


def get_questions_with_exemplars(
    info,
    n_shots,
    do_strong_shuffle,
    load_fn=load_hf_dataset
):

    # If ds_info is a function that tells us that the dataset
    # should be loaded using that custom function
    if callable(info):
        return info(n_shots=n_shots, do_strong_shuffle=do_strong_shuffle)

    # Create exemplars
    exemplar_ds = load_fn(
        path=info.path,
        name=info.name,
        data_dir=info.data_dir,
        split=info.exemplar_split
    )
    exemplars = [Exemplar(**info.extractor(row)) for row in exemplar_ds]
    random.seed(REPRODUCIBILITY_SEED)
    if do_strong_shuffle:
        for exemplar in exemplars:
            exemplar.strong_shuffle()

    # Create questions with exemplars
    eval_ds = load_fn(
        path=info.path,
        name=info.name,
        data_dir=info.data_dir,
        split=info.eval_split
    )

    random.seed(REPRODUCIBILITY_SEED)
    qwes = list()
    for row_idx, row in enumerate(eval_ds):

        # Choose some random exemplars - we are careful here
        # to avoid choosing an exemplar that is the same as
        # the question
        if info.exemplar_split == info.eval_split:
            possible_idxs = [i for i in range(len(exemplars)) if i != row_idx]
        else:
            possible_idxs = list(range(len(exemplars)))
        row_exemplars = [
            exemplars[i] for i in random.sample(possible_idxs, n_shots)
        ]

        row_qwe = QuestionWithExemplars(
            **{**info.extractor(row), **{"exemplars": row_exemplars}}
        )
        qwes.append(row_qwe)
    random.seed(REPRODUCIBILITY_SEED)
    if do_strong_shuffle:
        for qwe in qwes:
            qwe.strong_shuffle()

    return qwes


def load_tiny_obqa(n_shots, do_strong_shuffle):
    qwes = get_questions_with_exemplars(
        info=get_dataset_info("obqa"),
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 100)


def load_mini_rm(n_shots, do_strong_shuffle):
    qwes = get_questions_with_exemplars(
        info=get_dataset_info("rm"),
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 500)


def load_mini_sc(n_shots, do_strong_shuffle):
    qwes = get_questions_with_exemplars(
        info=get_dataset_info("sc"),
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 500)


def read_lqa(path, name, data_dir, split):

    with open(f"logiqa_{split}.txt", "r") as f:
        lines = f.readlines()

    grouper = groupby(lines, key=lambda x: x in {"\n"})
    ds = dict(enumerate((list(j) for i, j in grouper if not i), 1)).values()

    formatted_ds = list()
    for row in ds:
        formatted_row = dict()
        formatted_row["answer_idx"] = ord(row[0][0]) - ord("a")
        formatted_row["question"] = f"{row[1].strip()} {row[2].strip()}"
        choices = list()
        for choice in row[3:]:
            choices.append(choice[2:].strip())
        formatted_row["choices"] = choices
        formatted_ds.append(formatted_row)
    return formatted_ds


def load_lqa(n_shots, do_strong_shuffle):
    return get_questions_with_exemplars(
        info=DatasetInfo(
            path=None,
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["choices"],
                "answer_idx": row["answer_idx"]
            }
        ),
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle,
        load_fn=read_lqa
    )


def load_mmlu(n_shots, do_strong_shuffle):
    all_qwes = list()
    for name in MMLU_NAMES:
        name_qwes = get_questions_with_exemplars(
            info=DatasetInfo(
                path="hendrycks_test",
                name=name,
                exemplar_split="dev",
                eval_split="test",
                extractor=lambda row: {
                    "parts": [
                        QuestionPart(row["question"], tag="Question")
                    ],
                    "choices": row["choices"],
                    "answer_idx": (
                        row["answer"]
                        if isinstance(row["answer"], int)
                        else ltr_to_idx(row["answer"])
                    ),
                    "task": name
                }
            ),
            n_shots=n_shots,
            do_strong_shuffle=do_strong_shuffle,
            load_fn=load_hf_dataset_no_verify
        )
        all_qwes.extend(name_qwes)
    return all_qwes


def rm_final_period(text):
    return text[:-1] if text.endswith(".") else text


def get_anli_dataset_info(round):
    return DatasetInfo(
        path="anli",
        exemplar_split=f"train_r{round}",
        eval_split=f"test_r{round}",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["premise"], tag="Premise"),
                QuestionPart(text=row["hypothesis"], tag="Hypothesis")
            ],
            "choices": [
                "Hypothesis is definitely true given premise",
                "Hypothesis might be true given premise",
                "Hypothesis is definitely not true given premise"
            ],
            "answer_idx": row["label"]
        }
    )


def get_anli_shuffled_dataset_info(round):
    return DatasetInfo(
        path="anli",
        exemplar_split=f"train_r{round}",
        eval_split=f"test_r{round}",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["premise"], tag="Premise"),
                QuestionPart(text=row["hypothesis"], tag="Hypothesis")
            ],
            "choices": [
                "Hypothesis is definitely not true given premise",
                "Hypothesis is definitely true given premise",
                "Hypothesis might be true given premise"
            ],
            "answer_idx": {0: 1, 1: 2, 2: 0}[row["label"]]
        }
    )


def get_csqa_dataset_info(test):
    return DatasetInfo(
        path="commonsense_qa",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["question"], tag="Question")
            ],
            "choices": row["choices"]["text"],
            "answer_idx": None if row["answerKey"] == "" else (
                row["choices"]["label"].index(row["answerKey"])
            )
        }
    )


def get_siqa_dataset_info(test):
    return DatasetInfo(
        path="social_i_qa",
        exemplar_split="train",
        eval_split="socialiqa.jsonl" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=f"{row['context']} {row['question']}",
                    tag="Question"
                )
            ],
            "choices": [row[f"answer{idx_to_ltr(i)}"] for i in range(3)],
            "answer_idx": (
                int(row["label"]) - 1 if "label" in row.keys() else None
            )
        }
    )


def get_copa_dataset_info(test):
    return DatasetInfo(
        path="super_glue",
        name="copa",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=rm_final_period(row["premise"]) + (
                        " because" if row["question"] == "cause" else " so"
                    ),
                    tag="Question"
                )
            ],
            "choices": [row[f"choice{i+1}"] for i in range(2)],
            "answer_idx": row["label"]
        }
    )


def get_piqa_dataset_info(test):
    return DatasetInfo(
        path="piqa",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(row["goal"], tag="Question")
            ],
            "choices": [row[f"sol{i+1}"] for i in range(2)],
            "answer_idx": row["label"]
        }
    )


def get_cqa_dataset_info(test):
    return DatasetInfo(
        path="cosmos_qa",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=row["context"],
                    tag="Passage"
                ),
                QuestionPart(
                    text=row["question"],
                    tag="Question"
                )
            ],
            "choices": [row[f"answer{i}"] for i in range(4)],
            "answer_idx": row["label"]
        }
    )


def get_figqa_dataset_info(test):
    return DatasetInfo(
        path="nightingal3/fig-qa",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=f"{rm_final_period(row['startphrase'])}, meaning",
                    tag="Question"
                )
            ],
            "choices": [row[f"ending{i+1}"] for i in range(2)],
            "answer_idx": row["labels"]
        }
    )


def get_hs_dataset_info(test):

    return DatasetInfo(
        path="hellaswag",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    (
                        f"({row['activity_label']}) " if
                        row["source_id"].startswith("activity")
                        else ""
                    ) + row["ctx_a"],
                    tag="Passage"
                ),
                QuestionPart(
                    "Which choice best continues the passage?",
                    tag="Question"
                )
            ],
            "choices": [
                f"{row['ctx_b']}{' ' if len(row['ctx_b']) else ''}{e}"
                for e in row["endings"]
            ],
            "answer_idx": int(row["label"]) if len(row["label"]) else None
        }
    )


def get_medmcqa_dataset_info(test):
    return DatasetInfo(
        path="medmcqa",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(row["question"], tag="Question")
            ],
            "choices": [row[f"op{chr(i+ord('a'))}"] for i in range(4)],
            "answer_idx": row["cop"]
        }
    )


def get_rs_dataset_info(test):
    return DatasetInfo(
        path="riddle_sense",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=row["question"],
                    tag="Question"
                )
            ],
            "choices": row["choices"]["text"],
            "answer_idx": (
                None if row["answerKey"] == ""
                else row["choices"]["label"].index(row["answerKey"])
            )
        }
    )


def get_winogrande_dataset_info(test, xs):
    return DatasetInfo(
        path="winogrande",
        name="winogrande_xs" if xs else "winogrande_xl",
        exemplar_split="train",
        eval_split="test" if test else "validation",
        extractor=lambda row: {
            "parts": [
                QuestionPart(row["sentence"], tag="Question")
            ],
            "choices": [row[f"option{i+1}"] for i in range(2)],
            "answer_idx": (
                None if row["answer"] == "" else
                int(row["answer"]) - 1
            )
        }
    )


def do_caps_corrupt(s):
    new_s = ""
    for c in s:
        # If the character is a letter, flip a coin to decide whether to
        # capitalize it.
        if c.isalpha():
            new_s += c.upper() if random.random() < 0.5 else c.lower()
        else:
            new_s += c
    return new_s


def do_space_corrupt(s):
    words = s.split()
    new_words = []
    for w in words:
        if len(w) > 2:
            # Add a space at a random position
            pos = random.randint(0, len(w))
            new_words.append(w[:pos] + " " + w[pos:])
        else:
            new_words.append(w)
    return " ".join(new_words)


def get_corrupt_fn_by_name(name):
    if name == "caps":
        return do_caps_corrupt
    elif name == "space":
        return do_space_corrupt
    else:
        raise ValueError(f"Unknown corruption type {name}")


def get_obqa_corrupt_dataset_info(corruption_type):
    random.seed(REPRODUCIBILITY_SEED)
    corrupt_fn = get_corrupt_fn_by_name(name=corruption_type)
    return DatasetInfo(
        path="openbookqa",
        name="main",
        exemplar_split="train",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["question_stem"], tag="Question")
            ],
            "choices": [corrupt_fn(s) for s in row["choices"]["text"]],
            "answer_idx": row["choices"]["label"].index(row["answerKey"])
        }
    )


def load_mini_rm_caps_corrupt(n_shots, do_strong_shuffle):
    random.seed(REPRODUCIBILITY_SEED)
    corrupt_fn = get_corrupt_fn_by_name(name="caps")
    info = DatasetInfo(
        path="race",
        name="middle",
        exemplar_split="train",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["article"], tag="Passage"),
                QuestionPart(text=row["question"], tag="Question")
            ],
            "choices": [corrupt_fn(c) for c in row["options"]],
            "answer_idx": ltr_to_idx(row["answer"])
        }
    )
    qwes = get_questions_with_exemplars(
        info=info,
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 500)


def load_mini_rm_space_corrupt(n_shots, do_strong_shuffle):
    random.seed(REPRODUCIBILITY_SEED)
    corrupt_fn = get_corrupt_fn_by_name(name="space")
    info = DatasetInfo(
        path="race",
        name="middle",
        exemplar_split="train",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["article"], tag="Passage"),
                QuestionPart(text=row["question"], tag="Question")
            ],
            "choices": [corrupt_fn(c) for c in row["options"]],
            "answer_idx": ltr_to_idx(row["answer"])
        }
    )
    qwes = get_questions_with_exemplars(
        info=info,
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 500)


def load_mini_sc_caps_corrupt(n_shots, do_strong_shuffle):
    random.seed(REPRODUCIBILITY_SEED)
    corrupt_fn = get_corrupt_fn_by_name(name="caps")
    info = DatasetInfo(
        path="story_cloze",
        name="2016",
        data_dir="sc_data",
        exemplar_split="validation",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=" ".join(
                        [row[f"input_sentence_{i+1}"] for i in range(4)]
                    ),
                    tag="Story"
                ),
                QuestionPart(
                    text=(
                        "Which sentence best completes the story?"
                    ),
                    tag="Question"
                )
            ],
            "choices": [
                corrupt_fn(row[f"sentence_quiz{i+1}"]) for i in range(2)
            ],
            "answer_idx": row["answer_right_ending"] - 1
        }
    )
    qwes = get_questions_with_exemplars(
        info=info,
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 500)


def load_mini_sc_space_corrupt(n_shots, do_strong_shuffle):
    random.seed(REPRODUCIBILITY_SEED)
    corrupt_fn = get_corrupt_fn_by_name(name="space")
    info = DatasetInfo(
        path="story_cloze",
        name="2016",
        data_dir="sc_data",
        exemplar_split="validation",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(
                    text=" ".join(
                        [row[f"input_sentence_{i+1}"] for i in range(4)]
                    ),
                    tag="Story"
                ),
                QuestionPart(
                    text=(
                        "Which sentence best completes the story?"
                    ),
                    tag="Question"
                )
            ],
            "choices": [
                corrupt_fn(row[f"sentence_quiz{i+1}"]) for i in range(2)
            ],
            "answer_idx": row["answer_right_ending"] - 1
        }
    )
    qwes = get_questions_with_exemplars(
        info=info,
        n_shots=n_shots,
        do_strong_shuffle=do_strong_shuffle
    )
    random.seed(REPRODUCIBILITY_SEED)
    return random.sample(qwes, 500)


def get_mini_rm_corrupt_dataset_info(corruption_type):
    random.seed(REPRODUCIBILITY_SEED)
    corrupt_fn = get_corrupt_fn_by_name(name=corruption_type)
    return DatasetInfo(
        path="race",
        name="middle",
        exemplar_split="train",
        eval_split="test",
        extractor=lambda row: {
            "parts": [
                QuestionPart(text=row["article"], tag="Passage"),
                QuestionPart(text=row["question"], tag="Question")
            ],
            "choices": [corrupt_fn(c) for c in row["options"]],
            "answer_idx": ltr_to_idx(row["answer"])
        }
    )


def get_dataset_info(ds_name):
    return {
        "obqa": DatasetInfo(
            path="openbookqa",
            name="main",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question_stem"], tag="Question")
                ],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
            }
        ),
        "ae": DatasetInfo(
            path="ai2_arc",
            name="ARC-Easy",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
            }
        ),
        "ac": DatasetInfo(
            path="ai2_arc",
            name="ARC-Challenge",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["choices"]["text"],
                "answer_idx": row["choices"]["label"].index(row["answerKey"])
            }
        ),
        "csqa": get_csqa_dataset_info(test=False),
        "csqa_ts": get_csqa_dataset_info(test=True),
        "rh": DatasetInfo(
            path="race",
            name="high",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["article"], tag="Passage"),
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["options"],
                "answer_idx": ltr_to_idx(row["answer"])
            }
        ),
        "rm": DatasetInfo(
            path="race",
            name="middle",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["article"], tag="Passage"),
                    QuestionPart(text=row["question"], tag="Question")
                ],
                "choices": row["options"],
                "answer_idx": ltr_to_idx(row["answer"])
            }
        ),
        "siqa": get_siqa_dataset_info(test=False),
        "siqa_ts": get_siqa_dataset_info(test=True),
        "copa": get_copa_dataset_info(test=False),
        "copa_ts": get_copa_dataset_info(test=True),
        "figqa": get_figqa_dataset_info(test=False),
        "figqa_ts": get_figqa_dataset_info(test=True),
        "rs": get_rs_dataset_info(test=False),
        "rs_ts": get_rs_dataset_info(test=True),
        "agn": DatasetInfo(
            path="ag_news",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(row["text"], tag="Article"),
                    QuestionPart(
                        text=(
                            "What is the best classification "
                            "for this article?"
                        ),
                        tag="Question"
                    )
                ],
                "choices": ["World", "Sports", "Business", "Sci/Tech"],
                "answer_idx": row["label"]
            }
        ),
        "sc": DatasetInfo(
            path="story_cloze",
            name="2016",
            data_dir="sc_data",
            exemplar_split="validation",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        text=" ".join(
                            [row[f"input_sentence_{i+1}"] for i in range(4)]
                        ),
                        tag="Story"
                    ),
                    QuestionPart(
                        text=(
                            "Which sentence best completes the story?"
                        ),
                        tag="Question"
                    )
                ],
                "choices": [row[f"sentence_quiz{i+1}"] for i in range(2)],
                "answer_idx": row["answer_right_ending"] - 1
            }
        ),
        "a1": get_anli_dataset_info(1),
        "a2": get_anli_dataset_info(2),
        "a3": get_anli_dataset_info(3),
        "medmcqa": get_medmcqa_dataset_info(test=False),
        "medmcqa_ts": get_medmcqa_dataset_info(test=True),
        "dream": DatasetInfo(
            path="dream",
            exemplar_split="train",
            eval_split="test",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(" ".join(row["dialogue"]), tag="Dialogue"),
                    QuestionPart(row["question"], tag="Question")
                ],
                "choices": row["choice"],
                "answer_idx": row["choice"].index(row["answer"])
            }
        ),
        "codah": DatasetInfo(
            path="codah",
            name="codah",
            exemplar_split="train",
            eval_split="train",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(row["question_propmt"], tag="Question")
                ],
                "choices": row["candidate_answers"],
                "answer_idx": row["correct_answer_idx"]
            }
        ),
        "piqa": get_piqa_dataset_info(test=False),
        "piqa_ts": get_piqa_dataset_info(test=True),
        "w": get_winogrande_dataset_info(test=False, xs=False),
        "w_ts": get_winogrande_dataset_info(test=True, xs=False),
        "cqa": get_cqa_dataset_info(test=False),
        "cqa_ts": get_cqa_dataset_info(test=True),
        "mmlu": load_mmlu,
        "wxs": get_winogrande_dataset_info(test=False, xs=True),
        "wxs_ts": get_winogrande_dataset_info(test=True, xs=True),
        "tiny_obqa": load_tiny_obqa,
        "lqa": load_lqa,
        "hs": get_hs_dataset_info(test=False),
        "hs_ts": get_hs_dataset_info(test=True),
        "mini_rm": load_mini_rm,
        "mini_sc": load_mini_sc,
        "a1s": get_anli_shuffled_dataset_info(1),
        "a2s": get_anli_shuffled_dataset_info(2),
        "a3s": get_anli_shuffled_dataset_info(3),
        "obqa_caps_corrupt": get_obqa_corrupt_dataset_info(
            corruption_type="caps"
        ),
        "obqa_space_corrupt": get_obqa_corrupt_dataset_info(
            corruption_type="space"
        ),
        "mini_sc_caps_corrupt": load_mini_sc_caps_corrupt,
        "mini_sc_space_corrupt": load_mini_sc_space_corrupt,
        "mini_rm_caps_corrupt": load_mini_rm_caps_corrupt,
        "mini_rm_space_corrupt": load_mini_rm_space_corrupt
    }[ds_name]
