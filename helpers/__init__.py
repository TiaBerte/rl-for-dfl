import sys

LEGACY_MODULES = {"rl_utils": "usecases.ems.vpp_envs",
                  "rl.__init__": "usecases.setcover.rl.__init__",
                  "rl.algos": "usecases.setcover.rl.algos",
                  "rl.utility": "usecases.setcover.rl.utility"}
TIMESTEP_IN_A_DAY = 96
METHODS = ['hybrid-single-step', 'hybrid-mdp', 'rl-single-step', 'rl-mdp']
MODES = ['train', 'test']
MIN_REWARD = -10000
PYTHON_VERSION = sys.version_info

assert PYTHON_VERSION.major == 3, "Only Python 3.7 or 3.8 are supported"
# assert PYTHON_VERSION.minor in {7, 8}, "Only Python 3.7 or 3.8 are supported"

if PYTHON_VERSION.minor <= 7:
    import pickle5 as pickle
else:
    import pickle