"""
    Custom baselines.
"""

import numpy as np

from garage.np.baselines.baseline import Baseline

########################################################################################################################


class ZeroBaseline(Baseline):
    """
        This is a fake baseline and it simply returns 0.
    """

    def __init__(self, env_spec):
        self.name = 'ZeroBaseline'

    def get_param_values(self, **kwargs):
        return None

    def set_param_values(self, val, **kwargs):
        pass

    def fit(self, paths):
        pass

    def predict(self, path):
        return np.zeros(shape=(1,))

    def predict_n(self, paths):
        return [np.zeros(shape=(1, )) for path in paths]