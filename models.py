from constants import (
    CODEX_MODEL_NAME,
    CP_MODEL_NAME,
    CURIE_MODEL_NAME,
    GPT2_MODEL_NAME,
    GPT3_MODEL_NAME,
    HF_CACHE_DIR_NAME,
    INSTRUCT_MODEL_NAME,
    JURASSIC_MODEL_NAME,
    JURASSIC_SPACE,
    RETRY_SLEEP_TIME
)
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import idx_to_ltr, prep_openai_obj_for_save

import numpy as np
import openai
import requests
import time


@dataclass
class ModelResponseNatural:
    logprobs: dict
    response_list: list


@dataclass
class ModelResponseBrown:
    logprobs: dict
    unconditional_logprobs: dict
    lens: dict
    response_list: list


class Test:

    def _get_uniform_response(self, n_choices):
        return {idx_to_ltr(i): np.log(1/n_choices) for i in range(n_choices)}

    def process_question_natural(self, question):
        n_choices = question.get_n_choices()
        logprobs = self._get_uniform_response(n_choices=n_choices)
        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=list()
        )

    def process_question_brown(self, question):
        n_choices = question.get_n_choices()
        logprobs = self._get_uniform_response(n_choices=n_choices)
        lens = {idx_to_ltr(i): 1 for i in range(n_choices)}
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=logprobs,
            lens=lens,
            response_list=list()
        )


class GPT2Model:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR_NAME
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR_NAME
        )
        self.lbls_map = {v: k for k, v in self.tokenizer.vocab.items()}

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        probs = logits.softmax(dim=-1)
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }

        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:200]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            response_list=list()
        )

    def process_question_brown(self):
        pass


class CodeParrot(GPT2Model):

    def __init__(self):
        super().__init__(model_name=CP_MODEL_NAME)


class GPT2(GPT2Model):

    def __init__(self):
        super().__init__(model_name=GPT2_MODEL_NAME)


class Jurassic:

    def __init__(self, api_key):
        self.key = api_key

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()

        response = requests.post(
            f"https://api.ai21.com/studio/v1/{JURASSIC_MODEL_NAME}/complete",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "prompt": prompt_text,
                "numResults": 1,
                "maxTokens": 1,
                "topKReturn": 64,
                "temperature": 1.0,
            }
        )

        while True:
            resp_json = response.json()
            try:
                completion_tokens = resp_json["completions"][0]["data"]["tokens"][0]["topTokens"]
                break
            except Exception:
                print(resp_json)
                print(f"Will retry API call in {RETRY_SLEEP_TIME} seconds...")
                time.sleep(RETRY_SLEEP_TIME)

        log_probs = {t["token"]: t["logprob"] for t in completion_tokens}
        completion_tokens = {k.replace(JURASSIC_SPACE, " "): v
                             for k, v in log_probs.items()}
        return ModelResponseNatural(
            logprobs=completion_tokens,
            response_list=[resp_json]
        )

    def process_question_brown(self):
        pass


class OpenAIModel:

    def __init__(self, api_key, model_name, add_space=False):
        openai.api_key = api_key
        self.add_space = add_space
        self.model_name = model_name

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()
        response = self._get_response(text=prompt_text, echo=False)
        logprobs = dict(response["choices"][0]["logprobs"]["top_logprobs"][0])

        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=[
                prep_openai_obj_for_save(
                    obj=response,
                    prompt_text=prompt_text
                )
            ]
        )

    def _get_response(self, text, echo):
        while True:
            try:
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=text+(" " if self.add_space else ""),
                    temperature=0,  # Doesn't actually matter here
                    max_tokens=1,  # Just need to get letter
                    logprobs=5,  # Get max number of logprobs
                    echo=echo
                )
                return response
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)

    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()

        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()

        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)

            # Get unconditional logprobs
            response = self._get_response(text=f"Answer: {choice}", echo=True)
            choice_logprobs = (
                response["choices"][0]["logprobs"]["token_logprobs"][2:-1]
            )

            choice_n_tokens = len(choice_logprobs)
            unconditional_logprobs[ltr] = sum(choice_logprobs)
            lens[ltr] = choice_n_tokens
            response_list.append(
                prep_openai_obj_for_save(
                    obj=response,
                    prompt_text=f"Answer: {choice}"
                )
            )

            # Get conditional logprobs
            response = self._get_response(
                text=f"{prompt_text} {choice}", echo=True
            )
            token_logprobs = (
                response["choices"][0]["logprobs"]["token_logprobs"]
            )
            choice_logprobs = token_logprobs[-(choice_n_tokens+1):-1]

            logprobs[ltr] = sum(choice_logprobs)
            response_list.append(
                prep_openai_obj_for_save(
                    obj=response,
                    prompt_text=f"{prompt_text} {choice}"
                )
            )

        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )


class Codex(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(
            api_key=api_key,
            model_name=CODEX_MODEL_NAME,
            add_space=True
        )


class GPT3(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT3_MODEL_NAME)


class Instruct(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=INSTRUCT_MODEL_NAME)


class Curie(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=CURIE_MODEL_NAME)


def get_model_by_name(name, api_key):
    try:
        return {
            "codex": Codex,
            "gpt3": GPT3,
            "instruct": Instruct,
            "curie": Curie,
            "jurassic": Jurassic
        }[name](api_key=api_key)
    except KeyError:
        return {
            "test": Test,
            "cp": CodeParrot,
            "gpt2": GPT2
        }[name]()
