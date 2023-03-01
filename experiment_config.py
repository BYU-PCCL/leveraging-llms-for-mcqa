from constants import RESULTS_DIR_NAME
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    ds_name: str
    model_name: str
    style_name: str
    n_shots: int
    do_strong_shuffle: bool
    do_perm: bool

    def get_save_fname(self):
        vals = [str(v) for v in vars(self).values()]
        return f"{RESULTS_DIR_NAME}/{'_'.join(vals)}.pkl"
