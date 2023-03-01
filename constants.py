from utils import make_dir_if_does_not_exist


CODEX_MODEL_NAME = "code-davinci-002"
CP_MODEL_NAME = "codeparrot/codeparrot"
GPT2_MODEL_NAME = "gpt2"
GPT3_MODEL_NAME = "davinci"
CURIE_MODEL_NAME = "text-curie-001"  # Instruct
INSTRUCT_MODEL_NAME = "text-davinci-002"
JURASSIC_MODEL_NAME = "j1-jumbo"

JURASSIC_SPACE = "▁"
REPRODUCIBILITY_SEED = 0
RETRY_SLEEP_TIME = 30
SAVE_EVERY = 25

MMLU_NAMES = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]


HF_CACHE_DIR_NAME = "hf_cache"
RESULTS_DIR_NAME = "results"


for dir_name in [HF_CACHE_DIR_NAME, RESULTS_DIR_NAME]:
    make_dir_if_does_not_exist(dir_name)
