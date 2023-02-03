"""
    Methods that were modified to customize the logging and evaluation during training.
"""

import functools
import gc
import collections
import os
import subprocess
import dowel
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from dowel import tabular, logger
import time
import copy

from garage.np import discount_cumsum
from garage import StepType
from garage.trainer import Trainer, NotSetupError, TrainArgs
from garage import EnvSpec
from garage.envs.gym_env import GymEnv
from garage.experiment.experiment import ExperimentContext
from garage.experiment.experiment import dump_json, make_launcher_archive
from garage.experiment.experiment import _make_sequential_log_dir, _make_experiment_signature
from garage.experiment.deterministic import set_seed
from garage.tf.policies import Policy

import __main__ as main

########################################################################################################################


class CustomExperimentTemplate:
    """
    Custom implementation of garage.experiment.experiment.ExperimentTemplate to avoid environment serialization.
    (https://github.com/rlworkgroup/garage/blob/3492f446633a7e748f2f79077f6301c5b3ec9281/src/garage/experiment/experiment.py)

    Creates experiment log directories and runs an experiment.

    This class should only be created by calling garage.wrap_experiment.
    Generally, it's used as a decorator like this:

        @wrap_experiment(snapshot_mode='all')
        def my_experiment(ctxt, seed, lr=0.5):
            ...

        my_experiment(seed=1)

    Even though this class could be implemented as a closure in
    wrap_experiment(), it's more readable (and easier to pickle) implemented as
    a class.

    Note that the full path that will be created is
    f'{data}/local/{prefix}/{name}'.

    Args:
        function (callable or None): The experiment function to wrap.
        log_dir (str or None): The full log directory to log to. Will be
            computed from `name` if omitted.
        name (str or None): The name of this experiment template. Will be
            filled from the wrapped function's name if omitted.
        prefix (str): Directory under data/local in which to place the
            experiment directory.
        snapshot_mode (str): Policy for which snapshots to keep (or make at
            all). Can be either "all" (all iterations will be saved), "last"
            (only the last iteration will be saved), "gap" (every snapshot_gap
            iterations are saved), or "none" (do not save snapshots).
        snapshot_gap (int): Gap between snapshot iterations. Waits this number
            of iterations before taking another snapshot.
        archive_launch_repo (bool): Whether to save an archive of the
            repository containing the launcher script. This is a potentially
            expensive operation which is useful for ensuring reproducibility.
        name_parameters (str or None): Parameters to insert into the experiment
            name. Should be either None (the default), 'all' (all parameters
            will be used), or 'passed' (only passed parameters will be used).
            The used parameters will be inserted in the order they appear in
            the function definition.
        use_existing_dir (bool): If true, (re)use the directory for this
            experiment, even if it already contains data.
        x_axis (str): Key to use for x axis of plots.



    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *, function, log_dir, name, prefix, snapshot_mode,
                 snapshot_gap, archive_launch_repo, name_parameters,
                 use_existing_dir, x_axis):
        self.function = function
        self.log_dir = log_dir
        self.name = name
        self.prefix = prefix
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap
        self.archive_launch_repo = archive_launch_repo
        self.name_parameters = name_parameters
        self.use_existing_dir = use_existing_dir
        self.x_axis = x_axis
        if self.function is not None:
            self._update_wrap_params()

    def _update_wrap_params(self):
        """Update self to "look like" the wrapped funciton.

        Mostly, this involves creating a function signature for the
        ExperimentTemplate that looks like the wrapped function, but with the
        first argument (ctxt) excluded, and all other arguments required to be
        keyword only.

        """
        functools.update_wrapper(self, self.function)
        self.__signature__ = _make_experiment_signature(self.function)

    @classmethod
    def _augment_name(cls, options, name, params):
        """Augment the experiment name with parameters.

        Args:
            options (dict): Options to `wrap_experiment` itself. See the
                function documentation for details.
            name (str): Name without parameter names.
            params (dict): Dictionary of parameters.

        Raises:
            ValueError: If self.name_parameters is not set to None, "passed",
                or "all".

        Returns:
            str: Returns the augmented name.

        """
        name_parameters = collections.OrderedDict()

        if options['name_parameters'] == 'passed':
            for param in options['signature'].parameters.values():
                try:
                    name_parameters[param.name] = params[param.name]
                except KeyError:
                    pass
        elif options['name_parameters'] == 'all':
            for param in options['signature'].parameters.values():
                name_parameters[param.name] = params.get(
                    param.name, param.default)
        elif options['name_parameters'] is not None:
            raise ValueError('wrap_experiment.name_parameters should be set '
                             'to one of None, "passed", or "all"')
        param_str = '_'.join('{}={}'.format(k, v)
                             for (k, v) in name_parameters.items())
        if param_str:
            return '{}_{}'.format(name, param_str)
        else:
            return name

    def _get_options(self, *args):
        """Get the options for wrap_experiment.

        This method combines options passed to `wrap_experiment` itself and to
        the wrapped experiment.

        Args:
            args (list[dict]): Unnamed arguments to the wrapped experiment. May
                be an empty list or a list containing a single dictionary.

        Raises:
            ValueError: If args contains more than one value, or the value is
                not a dictionary containing at most the same keys as are
                arguments to `wrap_experiment`.

        Returns:
            dict: The final options.

        """
        options = dict(name=self.name,
                       function=self.function,
                       prefix=self.prefix,
                       name_parameters=self.name_parameters,
                       log_dir=self.log_dir,
                       archive_launch_repo=self.archive_launch_repo,
                       snapshot_gap=self.snapshot_gap,
                       snapshot_mode=self.snapshot_mode,
                       use_existing_dir=self.use_existing_dir,
                       x_axis=self.x_axis,
                       signature=self.__signature__)
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                for k in args[0]:
                    if k not in options:
                        raise ValueError('Unknown key {} in wrap_experiment '
                                         'options'.format(k))
                options.update(args[0])
            else:
                raise ValueError('garage.experiment currently only supports '
                                 'keyword arguments')
        return options

    @classmethod
    def _make_context(cls, options, **kwargs):
        """Make a context from the template information and variant args.

        Currently, all arguments should be keyword arguments.

        Args:
            options (dict): Options to `wrap_experiment` itself. See the
                function documentation for details.
            kwargs (dict): Keyword arguments for the wrapped function. Will be
                logged to `variant.json`

        Returns:
            ExperimentContext: The created experiment context.

        """
        name = options['name']
        if name is None:
            name = options['function'].__name__
        name = cls._augment_name(options, name, kwargs)
        log_dir = options['log_dir']
        if log_dir is None:
            log_dir = ('{data}/local/{prefix}/{name}'.format(
                data=os.path.join(os.getcwd(), 'data'),
                prefix=options['prefix'],
                name=name))
        if options['use_existing_dir']:
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = _make_sequential_log_dir(log_dir)

        tabular_log_file = os.path.join(log_dir, 'progress.csv')
        text_log_file = os.path.join(log_dir, 'debug.log')
        variant_log_file = os.path.join(log_dir, 'variant.json')
        metadata_log_file = os.path.join(log_dir, 'metadata.json')

        kwargs_to_dump = kwargs.copy()
        # Since it may require a lot of memory, remove the environment from the serialization
        if 'env' in kwargs_to_dump.keys():
            del kwargs_to_dump['env']
        dump_json(variant_log_file, kwargs_to_dump)
        git_root_path, metadata = get_metadata()
        dump_json(metadata_log_file, metadata)
        if git_root_path and options['archive_launch_repo']:
            make_launcher_archive(git_root_path=git_root_path, log_dir=log_dir)

        logger.add_output(dowel.TextOutput(text_log_file))
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(
            dowel.TensorBoardOutput(log_dir, x_axis=options['x_axis']))
        logger.add_output(dowel.StdOutput())

        logger.push_prefix('[{}] '.format(name))
        logger.log('Logging to {}'.format(log_dir))

        return ExperimentContext(snapshot_dir=log_dir,
                                 snapshot_mode=options['snapshot_mode'],
                                 snapshot_gap=options['snapshot_gap'])

    def __call__(self, *args, **kwargs):
        """Wrap a function to turn it into an ExperimentTemplate.

        Note that this docstring will be overriden to match the function's
        docstring on the ExperimentTemplate once a function is passed in.

        Args:
            args (list): If no function has been set yet, must be a list
                containing a single callable. If the function has been set, may
                be a single value, a dictionary containing overrides for the
                original arguments to `wrap_experiment`.
            kwargs (dict): Arguments passed onto the wrapped function.

        Returns:
            object: The returned value of the wrapped function.

        Raises:
            ValueError: If not passed a single callable argument.

        """
        if self.function is None:
            if len(args) != 1 or len(kwargs) != 0 or not callable(args[0]):
                raise ValueError('Please apply the result of '
                                 'wrap_experiment() to a single function')
            # Apply ourselves as a decorator
            self.function = args[0]
            self._update_wrap_params()
            return self
        else:
            ctxt = self._make_context(self._get_options(*args), **kwargs)
            result = self.function(ctxt, **kwargs)
            logger.remove_all()
            logger.pop_prefix()
            gc.collect()  # See dowel issue #44
            return result

########################################################################################################################


class CustomTrainer(Trainer):
    """
    Custom implementation of the garage.trainer.Trainer to prevent environement serialization.
    (https://github.com/rlworkgroup/garage/blob/3492f446633a7e748f2f79077f6301c5b3ec9281/src/garage/trainer.py)
    """
    def __init__(self, snapshot_config):
        super().__init__(snapshot_config=snapshot_config)

    def train(self,
              n_epochs,
              batch_size=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False):
        """Start training.
        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.
        Raises:
            NotSetupError: If train() is called before setup().
        Returns:
            float: The average return in last epoch cycle.
        """
        if not self._has_setup:
            raise NotSetupError(
                'Use setup() to setup trainer before training.')

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot
        self._start_worker()

        average_return = self._algo.train(self)
        self._shutdown_worker()

        return average_return

    def save(self, epoch):
        """Save snapshot of current batch.
        Args:
            epoch (int): Epoch.
        Raises:
            NotSetupError: if save() is called before the trainer is set up.
        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['seed'] = self._seed
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        params['algo'] = self._algo
        params['params'] = self._algo.policy.parameters
        # params['n_workers'] = self._n_workers
        # params['worker_class'] = self._worker_class
        # params['worker_args'] = self._worker_args

        start = time.time()
        self._snapshotter.save_itr_params(epoch, params)
        end = time.time()
        print(end - start)

        logger.log('Saved')

    def restore(self, from_dir, env, val_env, from_epoch='last'):
        """Restore experiment from snapshot.
        Args:
            from_dir (str): Directory of the pickle file
                to resume experiment from.
            from_epoch (str or int): The epoch to restore from.
                Can be 'first', 'last' or a number.
                Not applicable when snapshot_mode='last'.
        Returns:
            TrainArgs: Arguments for train().
        """
        saved = self._snapshotter.load(from_dir, from_epoch)

        self._seed = saved['seed']
        self._train_args = saved['train_args']
        self._stats = saved['stats']

        set_seed(self._seed)

        # FIXME: algo is a policy not an garage algo
        self.setup(env=env, algo=saved['algo'], val_env=val_env)

        n_epochs = self._train_args.n_epochs
        last_epoch = self._stats.total_epoch
        last_itr = self._stats.total_itr
        total_env_steps = self._stats.total_env_steps
        batch_size = self._train_args.batch_size
        store_episodes = self._train_args.store_episodes
        pause_for_plot = self._train_args.pause_for_plot

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('-- Train Args --', '-- Value --'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_episodes', store_episodes))
        logger.log(fmt.format('pause_for_plot', pause_for_plot))
        logger.log(fmt.format('-- Stats --', '-- Value --'))
        logger.log(fmt.format('last_itr', last_itr))
        logger.log(fmt.format('total_env_steps', total_env_steps))

        self._train_args.start_epoch = last_epoch + 1
        return copy.copy(self._train_args)

########################################################################################################################


class CustomTFTrainer(CustomTrainer):
    """
    Custom implementation of garage.trainer.TFTrainer to prevent environment serialization.
    (https://github.com/rlworkgroup/garage/blob/3492f446633a7e748f2f79077f6301c5b3ec9281/src/garage/trainer.py)
    This class implements a trainer for TensorFlow algorithms.
    A trainer provides a default TensorFlow session using python context.
    This is useful for those experiment components (e.g. policy) that require a
    TensorFlow session during construction.
    Use trainer.setup(algo, env) to setup algorithm and environment for trainer
    and trainer.train() to start training.
    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by Trainer to create the snapshotter.
            If None, it will create one with default settings.
        sess (tf.Session): An optional TensorFlow session.
              A new session will be created immediately if not provided.
    Note:
        When resume via command line, new snapshots will be
        saved into the SAME directory if not specified.
        When resume programmatically, snapshot directory should be
        specify manually or through @wrap_experiment interface.
    Examples:
        # to train
        with TFTrainer() as trainer:
            env = gym.make('CartPole-v1')
            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32))
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                max_episode_length=100,
                discount=0.99,
                max_kl_step=0.01)
            trainer.setup(algo, env)
            trainer.train(n_epochs=100, batch_size=4000)
        # to resume immediately.
        with TFTrainer() as trainer:
            trainer.restore(resume_from_dir)
            trainer.resume()
        # to resume with modified training arguments.
        with TFTrainer() as trainer:
            trainer.restore(resume_from_dir)
            trainer.resume(n_epochs=20)
    """

    def __init__(self, snapshot_config, sess=None):
        # pylint: disable=import-outside-toplevel
        import tensorflow
        # pylint: disable=global-statement
        global tf
        tf = tensorflow
        super().__init__(snapshot_config=snapshot_config)
        self.sess = sess or tf.compat.v1.Session()
        self.sess_entered = False

    def __enter__(self):
        """Set self.sess as the default session.
        Returns:
            TFTrainer: This trainer.
        """
        if tf.compat.v1.get_default_session() is not self.sess:
            self.sess.__enter__()
            self.sess_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave session.
        Args:
            exc_type (str): Type.
            exc_val (object): Value.
            exc_tb (object): Traceback.
        """
        if tf.compat.v1.get_default_session(
        ) is self.sess and self.sess_entered:
            self.sess.__exit__(exc_type, exc_val, exc_tb)
            self.sess_entered = False

    # NOTE: make the environment accessible from the outside
    @property
    def env(self):
        return self._env

    # NOTE: make the validation environment accessible from the outside
    @property
    def val_env(self):
        return self._val_env

    def setup(self, algo, env, val_env=None):
        """Set up trainer and sessions for algorithm and environment.
        This method saves algo and env within trainer and creates a sampler,
        and initializes all uninitialized variables in session.
        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().
        Args:
            algo (RLAlgorithm): An algorithm instance.
            env (Environment): An environment instance.
            val_env (Environment): A validation environment instance.
        """
        self.initialize_tf_vars()
        logger.log(self.sess.graph)
        super().setup(algo, env)
        # NOTE: added the validation environment
        self._val_env = val_env

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        self._sampler.start_worker()
        if self._plot:
            # pylint: disable=import-outside-toplevel
            from garage.tf.plotter import Plotter
            self._plotter = Plotter(self.get_env_copy(),
                                    self._algo.policy,
                                    sess=tf.compat.v1.get_default_session())
            self._plotter.start()

    def initialize_tf_vars(self):
        """Initialize all uninitialized variables in session."""
        with tf.name_scope('initialize_tf_vars'):
            uninited_set = [
                e.decode() for e in self.sess.run(
                    tf.compat.v1.report_uninitialized_variables())
            ]
            self.sess.run(
                tf.compat.v1.variables_initializer([
                    v for v in tf.compat.v1.global_variables()
                    if v.name.split(':')[0] in uninited_set
                ]))

########################################################################################################################


@dataclass(frozen=True, init=False)
class CustomEnvSpec(EnvSpec):
    """
       This class extends the garage.EnvSpec class adding a demands scaler attribute.
       Describes the observations, actions, and time horizon of an MDP.
       Args:
           observation_space (akro.Space): The observation space of the env.
           action_space (akro.Space): The action space of the env.
           max_episode_length (int): The maximum number of steps allowed in an
               episode.
       """
    def __init__(self,
                 observation_space,
                 action_space,
                 scaler,
                 max_episode_length=None):

        object.__setattr__(self, 'scaler', scaler)

        super().__init__(action_space=action_space,
                         observation_space=observation_space,
                         max_episode_length=max_episode_length)

        scaler: StandardScaler

########################################################################################################################


class CustomEnv(GymEnv):
    """
    This class extends the garage.envs.GymEnv class adding the demands scaler attribute.
    """
    def __init__(self, env, is_image=False, max_episode_length=None):

        assert hasattr(env, 'demands_scaler'), "The environvment must have a demands_scaler attribute"

        super().__init__(env=env, is_image=is_image, max_episode_length=max_episode_length)

        self._spec = CustomEnvSpec(action_space=self.action_space,
                                   observation_space=self.observation_space,
                                   max_episode_length=self._max_episode_length,
                                   scaler=env.demands_scaler)

    @property
    def env(self):
        return self._env

########################################################################################################################


# FIXME: this method will not work if the environment has no multiple instances
def _evaluate_during_training(env: CustomEnv, policy: Policy, prefix: str):
    """
    Evaluate the policy during training on the environment given as input.
    :param env: CustomEnv; the environment on which the policy is evaluated.
    :param policy: garage.tf.policies.Policy; the policy to evaluate.
    :param prefix: str; the prefix used in the logger.
    :return:
    """

    # Save all the regrets and rewards.
    regrets = list()
    rewards = list()

    # Randomly choose a set of instances
    for idx in range(len(env.input_features)):

        # Episode variables initialization
        env.current_instance = idx
        done = False
        episode_rew = 0
        episode_regret = 0
        s_t = env.input_features[idx]

        # Perform an episode
        while not done:
            s_t = np.expand_dims(s_t, axis=0)

            # Choose the action
            _, action = policy.get_actions(s_t)
            a_t = action['mean']
            a_t = np.squeeze(a_t)

            # Act on the environment
            s_t, rew, done, info = env.step(action=a_t)

            # Keep track of results
            episode_rew += rew
            episode_regret += info['regret']

        regrets.append(episode_regret)
        rewards.append(episode_rew)

    # Display the average regret and reward
    avg_regret = np.mean(regrets)
    avg_reward = np.mean(rewards)

    with tabular.prefix(prefix + '/'):
        tabular.record('AvgRegret', avg_regret)
        tabular.record('AvgReward', avg_reward)

########################################################################################################################


def log_performance(itr,
                    batch,
                    validate_every_n_steps,
                    train_env,
                    val_env,
                    policy,
                    discount,
                    prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.
       Custom implementation of the log_performance method of garage
       (https://github.com/rlworkgroup/garage/blob/master/src/garage/_functions.py).
    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        validate_every_n_steps (int): Validation frequency.
        train_env (GymEnv): The environment used for training.
        val_env (GymEnv): The environment used for validation.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.
    Returns:
        numpy.ndarray: Undiscounted returns.
    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []

    # Initialize auxiliary metrics
    batch_feasibles = list()
    batch_true_cost = list()
    batch_regret = list()

    # If required, validate the agent on separate validation set and evaluate it on the training env
    if itr % validate_every_n_steps == 0:
        if val_env is not None:
            _evaluate_during_training(env=val_env, policy=policy, prefix='Validation')
            _evaluate_during_training(env=train_env, policy=policy, prefix='Training')

    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))

        # FIXME: this code is ugly...
        assert 'feasible' in eps.env_infos.keys(), "A feasible value is expected"
        assert 'true cost' in eps.env_infos.keys(), "A true cost value is expected"
        assert 'regret' in eps.env_infos.keys(), "A regret value is expected"

        successful = np.sum(~eps.env_infos['feasible']) == 0
        batch_feasibles.append(successful)
        batch_true_cost.append(eps.env_infos['true cost'][-1])
        batch_regret.append(eps.env_infos['regret'][-1])

        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])
    average_feasibility_ratio = np.mean(batch_feasibles)
    average_true_cost = np.mean(batch_true_cost)
    average_regret = np.mean(batch_regret)

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))

        # NOTE: add additional evaluation metrics
        tabular.record('BatchAvgFeasibilityRatio', average_feasibility_ratio)
        tabular.record('BatchAvgTrueCost', average_true_cost)
        tabular.record('BatchAvgRegret', average_regret)

        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns

########################################################################################################################


def get_metadata():
    """
    This custom implementation fix a bug for Windows OS system.
    Get metadata about the main script.

    The goal of this function is to capture the additional information needed
    to re-run an experiment, assuming that the launcher script that started the
    experiment is located in a clean git repository.

    Returns:
        tuple[str, dict[str, str]]:
          * Absolute path to root directory of launcher's git repo.
          * Directory containing:
            * githash (str): Hash of the git revision of the repo the
                experiment was started from. "-dirty" will be appended to this
                string if the repo has uncommitted changes. May not be present
                if the main script is not in a git repo.
            * launcher (str): Relative path to the main script from the base of
                the repo the experiment was started from. If the main script
                was not started from a git repo, this will instead be an
                absolute path to the main script.

    """
    main_file = getattr(main, '__file__', None)
    if not main_file:
        return None, {}
    main_file_path = os.path.abspath(main_file)
    try:
        git_root_path = subprocess.check_output(
            ('git', 'rev-parse', '--show-toplevel'),
            cwd=os.path.dirname(main_file_path),
            stderr=subprocess.DEVNULL)
        git_root_path = git_root_path.strip()
    except subprocess.CalledProcessError:
        # This file is always considered not to exist.
        git_root_path = ''
    # We check that the path exists since in old versions of git the above
    # rev-parse command silently exits with 0 when run outside of a git repo.
    if not os.path.exists(git_root_path):
        return None, {
            'launcher': main_file_path,
        }
    launcher_path = os.path.relpath(bytes(main_file_path, encoding='utf8'),
                                    git_root_path)

    # NOTE: fixed bug for Windows OS
    if os.name == 'nt':
        git_root_path = git_root_path.decode("utf-8")

    git_hash = subprocess.check_output(('git', 'rev-parse', 'HEAD'),
                                       cwd=git_root_path)
    git_hash = git_hash.decode('utf-8').strip()
    git_status = subprocess.check_output(('git', 'status', '--short'),
                                         cwd=git_root_path)
    git_status = git_status.decode('utf-8').strip()
    if git_status != '':
        git_hash = git_hash + '-dirty'
    return git_root_path, {
        'githash': git_hash,
        'launcher': launcher_path.decode('utf-8'),
    }

########################################################################################################################


def my_wrap_experiment(function,
                       logging_dir,
                       snapshot_mode,
                       snapshot_gap,
                       *,
                       prefix='experiment',
                       name=None,
                       archive_launch_repo=True,
                       name_parameters=None,
                       use_existing_dir=True,
                       x_axis='TotalEnvSteps'):
    """
    Custom wrapper for the ExperimentTemplate class of the garage library that allows to set the log directory.
    See the ExperimentTemplate class for more details.
    """
    return CustomExperimentTemplate(function=function,
                                    log_dir=logging_dir,
                                    prefix=prefix,
                                    name=name,
                                    snapshot_mode=snapshot_mode,
                                    snapshot_gap=snapshot_gap,
                                    archive_launch_repo=archive_launch_repo,
                                    name_parameters=name_parameters,
                                    use_existing_dir=use_existing_dir,
                                    x_axis=x_axis)
