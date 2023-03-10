import numpy as np
from usecases.wsmc.generate_instances import generate_training_and_test_sets
from garage.tf.baselines import ContinuousMLPBaseline
from garage.sampler import LocalSampler
from garage.tf.policies import GaussianMLPPolicy
from garage.experiment import SnapshotConfig

from usecases.wsmc.generate_instances import MinSetCoverEnv
from rl.algos import VPG
from helpers.garage_utility import CustomTFTrainer, CustomEnv, my_wrap_experiment

import yaml
import os
import tensorflow as tf

########################################################################################################################


def train(ctxt: SnapshotConfig = None,
          num_epochs: int = 100,
          batch_size: int = 100,
          env: MinSetCoverEnv = None):
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

    # A trainer provides a default TensorFlow session using python context
    with CustomTFTrainer(snapshot_config=ctxt) as trainer:

        # Garage wrapping of a gym environment
        env = CustomEnv(env, max_episode_length=1)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)
        obs, _ = env.reset()

        # A value function using a MLP network.
        baseline = ContinuousMLPBaseline(env_spec=env.spec)

        # It's called the "Local" sampler because it runs everything in the same process and thread as where
        # it was called from.
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=1,
                               is_tf_worker=True)

        # Vanilla Policy Gradient
        algo = VPG(env_spec=env.spec,
                   baseline=baseline,
                   policy=policy,
                   sampler=sampler,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.001, ),
                   center_adv=False)

        trainer.setup(algo, env)
        trainer.train(n_epochs=num_epochs, batch_size=batch_size, plot=False)

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
    # Generate training and test set in the specified directory
    generate_training_and_test_sets(data_path=DATA_PATH,
                                    num_instances=NUM_INSTANCES,
                                    num_sets=NUM_SETS,
                                    num_prods=NUM_PRODS,
                                    density=DENSITY,
                                    min_lmbd=MIN_LMBD,
                                    max_lmbd=MAX_LMBD)

    # Set the hyper parameters
    BATCH_SIZE = 100
    EPOCHS = 5000

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
    config_params = {'batch_size': BATCH_SIZE, 'epochs': EPOCHS}
    with open(os.path.join(SAVEPATH, 'config.yaml'), 'w') as file:
        yaml.dump(config_params, file)

    # Train and test the RL algo
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    run = my_wrap_experiment(train,
                             logging_dir=SAVEPATH,
                             snapshot_mode='gap_overwrite',
                             snapshot_gap=EPOCHS // 10,
                             # FIXME: archive_launch_repo=True is not supported
                             archive_launch_repo=False)
    run(num_epochs=EPOCHS, batch_size=BATCH_SIZE, env=env)