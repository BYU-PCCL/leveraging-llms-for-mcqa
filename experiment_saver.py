from collections import defaultdict

import pandas as pd


class ExperimentSaver(defaultdict):

    def __init__(self, save_fname):
        super().__init__(list)
        self.save_fname = save_fname

    def save(self):
        print("Saving to", self.save_fname)
        pd.DataFrame(self).to_pickle(self.save_fname)
