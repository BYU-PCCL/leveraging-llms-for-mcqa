import openai
import os


def idx_to_ltr(idx):
    return chr(idx + ord("A"))


def ltr_to_idx(ltr):
    return ord(ltr) - ord("A")


def make_dir_if_does_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def prep_openai_obj_for_save(obj, prompt_text=None):
    obj = dict(obj)
    for key in obj.keys():
        if isinstance(obj[key], openai.openai_object.OpenAIObject):
            obj[key] = prep_openai_obj_for_save(obj[key])
        if isinstance(obj[key], list):
            for i in range(len(obj[key])):
                if isinstance(obj[key][i], openai.openai_object.OpenAIObject):
                    obj[key][i] = prep_openai_obj_for_save(obj[key][i])
    if prompt_text is not None:
        obj["prompt_text"] = prompt_text
    return obj
