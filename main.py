import sklearn
import ortools
import numpy as np
from usecases.wsmc.generate_instances import generate_training_and_test_sets
from garage.sampler import LocalSampler
from garage.sampler import FragmentWorker
from garage.torch.policies.tanh_gaussian_mlp_policy import TanhGaussianMLPPolicy
from garage.torch.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from garage.experiment import SnapshotConfig
from usecases.wsmc.generate_instances import MinSetCoverEnv
from rl.algos import SAC
from garage.torch import set_gpu_mode
from garage.trainer import Trainer
from helpers.garage_utility import CustomEnv, my_wrap_experiment
from garage.replay_buffer.path_buffer import PathBuffer
import yaml
import os
import torch
import torch.nn as nn
from garage.replay_buffer import PathBuffer
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from prioritized_experience_replay import PrioritizedReplay



########################################################################################################################


from argparse import ArgumentParser
from typing import List

parser = ArgumentParser()

# Algorithm hyper-parameters
parser.add_argument("--batch_size", help="batch size", default=256, type=int)
parser.add_argument("--num_epochs", help="Number of epochs", default=500, type=int)
parser.add_argument("--hidden_dim", help="Q networks hidden dim", default=256, type=int)
parser.add_argument("--policy_lr", help="Learning rate", default=3e-3, type=float)
parser.add_argument("--qnet_lr", help="Learning rate", default=3e-2, type=float)
parser.add_argument("--optim", help="Optimizer", default="Adam", choices=["SGD", "Adam"], type=str)

# Experience replay hyper-parameters
parser.add_argument("--per", help="Using Prioritized Experience Replay", action='store_true')
parser.add_argument("--beta_start", help="Starting value of beta for prioritizing experience replay", default=0.4, type=float)
parser.add_argument("--annealing_rate", help="Annelaing rate of beta for prioritizing experience replay", default=0.4, type=float)
parser.add_argument("--alpha", help="Alpha value for prioritizing experience replay", default=0.6, type=float)

args = parser.parse_args()

args = parser.parse_args(["--batch_size", "1024", "--hidden_dim", "512",  "--per"])



def train(ctxt: SnapshotConfig = None,
          env: MinSetCoverEnv = None,
          args: ArgumentParser = None, 
          env_step_episode = 100):
    """
    :param ctxt: garage.experiment.SnapshotConfig: The snapshot configuration used by Trainer to create the
                                                   snapshotter.
    :param num_epochs: int; number of training epochs.
    :param batch_size: int; batch size.
    :param env: usecases.setcover.generate_instances.MinSetCoverEnv; the Minimum Set Cover environment instance.
    :return:
    """

    # Check that the env is not None
    assert env is not None

    # Garage wrapping of a gym environment
    env = CustomEnv(env, max_episode_length=1)
    env.observation_space.dtype = np.float64

    # Replay Buffer used for storing past experience
    if not args.per :
        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    else:
        replay_buffer = PrioritizedReplay(capacity_in_transitions=int(1e6),
                                          alpha=args.alpha,
                                          beta_start=args.beta_start,
                                          annealing_rate=args.annealing_rate)
    obs, _ = env.reset()


    # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[args.hidden_dim, args.hidden_dim],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    if args.optim == 'SGD':
        optim = torch.optim.SGD
    else:
        optim = torch.optim.Adam
    
    
    # Q-Network used as critic
    qf1 = ContinuousMLPQFunction(env.spec, hidden_sizes=[args.hidden_dim, args.hidden_dim])
    qf2 = ContinuousMLPQFunction(env.spec, hidden_sizes=[args.hidden_dim, args.hidden_dim])

    # It's called the "Local" sampler because it runs everything in the same process and thread as where
    # it was called from.
    sampler = LocalSampler(agents=policy,
                            envs=env,
                            max_episode_length=1,
                            worker_class=FragmentWorker)

    # Soft Actor Critic algorithm
    algo = SAC(env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                replay_buffer=replay_buffer,
                sampler=sampler,
                optimizer=optim,
                gradient_steps_per_itr=1,
                policy_lr=args.policy_lr,
                qf_lr=args.qnet_lr,
                per = args.per,
                buffer_batch_size=args.batch_size,
                )


    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    
    algo.to()

    trainer = Trainer(snapshot_config=ctxt)

    trainer.setup(algo=algo, env=env)
    trainer.train(n_epochs=args.num_epochs, batch_size=env_step_episode)#, plot=False)

########################################################################################################################


if __name__ == '__main__':
    # Min possible value for the Poisson rates
    MIN_LMBD = 1
    # Max possible value for the Poisson rates
    MAX_LMBD = 10
    # Number of products (elements)
    NUM_PRODS = 5
    # Number of sets (molds)
    NUM_SETS = 25
    # Density of the availability matrix
    DENSITY = 0.02
    # Number of instances to generate
    NUM_INSTANCES = 1000
    # Seed to ensure reproducible results
    SEED = 0
    DATA_PATH = os.path.join('data',
                             'wsmc',
                             f'{NUM_PRODS}x{NUM_SETS}',
                             'linear',
                             f'{NUM_INSTANCES}-instances',
                             f'seed-{SEED}')

    # Set the random seed to ensure reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Generate training and test set in the specified directory
    
    """
    generate_training_and_test_sets(data_path=DATA_PATH,
                                    num_instances=NUM_INSTANCES,
                                    num_sets=NUM_SETS,
                                    num_prods=NUM_PRODS,
                                    density=DENSITY,
                                    min_lmbd=MIN_LMBD,
                                    max_lmbd=MAX_LMBD)
    """

    # Set the hyper parameters
    # BATCH_SIZE = 100
    # EPOCHS = 3000

    SAVEPATH = os.path.join('models',
                            f'{NUM_PRODS}x{NUM_SETS}',
                            f'{NUM_INSTANCES}-instances',
                            f'seed-{SEED}')

    # Create the environment

    env = MinSetCoverEnv(num_prods=NUM_PRODS,
                         num_sets=NUM_SETS,
                         instances_filepath=DATA_PATH,
                         seed=SEED)
    

    # Create the saving directory if it does not exist
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)

    # Save the configuration params
    config_params = {"epochs": args.num_epochs,
                     "batch_size": args.batch_size,
                     "hidden_dim": args.hidden_dim,
                     "policy_lr": args.policy_lr,
                     "qnet_lr" : args.qnet_lr,
                     "optim": args.optim,
                     }

    if args.per:
        config_params["beta_start"]  = args.beta_start
        config_params["annealing_rate"] = args.annealing_rate
        config_params["alpha"] = args.alpha
    
    with open(os.path.join(SAVEPATH, 'config.yaml'), 'w') as file:
        yaml.dump(config_params, file)

    # Train and test the RL algo
    run = my_wrap_experiment(train,
                             logging_dir=SAVEPATH,
                             snapshot_mode='gap_overwrite',
                             snapshot_gap=args.num_epochs // 10,
                             # FIXME: archive_launch_repo=True is not supported
                             archive_launch_repo=False)
    
    run(env=env, args=args)