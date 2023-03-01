from dataclasses import dataclass
from utils import idx_to_ltr

import random


@dataclass
class QuestionPart:
    text: str
    tag: str = None

    def __str__(self):
        if self.tag is not None:
            return f"{self.tag}: {self.text}"
        else:
            return self.text


@dataclass
class Question:
    parts: list
    choices: list
    answer_idx: int
    task: str = None

    def get_n_choices(self):
        return len(self.choices)

    def get_answer_str(self):
        return self.choices[self.answer_idx]

    def _get_prompt(self, include_choices):
        prompt = ""
        for part in self.parts:
            prompt += f"{str(part)}\n"
        if include_choices:
            for i, choice in enumerate(self.choices):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        return prompt + "Answer:"

    def get_natural_prompt(self):
        return self._get_prompt(include_choices=True)

    def get_brown_prompt(self):
        return self._get_prompt(include_choices=False)

    def strong_shuffle(self):
        # This method shuffles choices such that choosing
        # the answer at the originally correct
        # index will mean getting the question wrong

        # For degenerate questions where all choices are the same
        if len(set(self.choices)) == 1:
            return

        answer_idx = self.answer_idx
        answer_str = self.get_answer_str()
        while self.choices[answer_idx] == answer_str:
            random.shuffle(self.choices)
            self.answer_idx = self.choices.index(answer_str)

    def permute_choices(self, perm):
        self.choices = [self.choices[i] for i in perm]
        self.answer_idx = perm.index(self.answer_idx)


class QuestionWithExemplars(Question):

    def __init__(self, parts, choices, answer_idx, exemplars, task=None):
        super().__init__(parts, choices, answer_idx, task)
        self.exemplars = exemplars

    def get_natural_prompt(self):
        prompt = super().get_natural_prompt()
        if len(self.exemplars):
            exemplar_prompts = [e.get_natural_prompt() for e in self.exemplars]
            exemplars = "\n\n".join(exemplar_prompts)
            return f"{exemplars}\n\n{prompt}"
        else:
            return prompt

    def get_brown_prompt(self):
        prompt = super().get_brown_prompt()
        if len(self.exemplars):
            exemplar_prompts = [e.get_brown_prompt() for e in self.exemplars]
            exemplars = "\n\n".join(exemplar_prompts)
            return f"{exemplars}\n\n{prompt}"
        else:
            return prompt


class Exemplar(Question):

    def get_natural_prompt(self):
        prompt = super().get_natural_prompt()
        answer_ltr = idx_to_ltr(self.answer_idx)
        return f"{prompt} {answer_ltr}"

    def get_brown_prompt(self):
        prompt = super().get_brown_prompt()
        return f"{prompt} {self.get_answer_str()}"
