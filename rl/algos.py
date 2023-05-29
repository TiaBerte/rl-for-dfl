"""Natural Policy Gradient Optimization implementation from garage
(https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/npo.py)."""
# pylint: disable=wrong-import-order
# yapf: disable
import collections
import time
from dowel import logger, tabular
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from garage import make_optimizer
from garage.np import explained_variance_1d, pad_batch_array
from garage.np.algos import RLAlgorithm
from garage.tf import (center_advs, compile_function, compute_advantages,
                       discounted_returns, flatten_inputs, graph_inputs,
                       positive_advs)
from garage.tf.optimizers import FirstOrderOptimizer
from garage.torch.algos.sac import SAC as sac
from garage.torch.algos.ppo import PPO as ppo

# NOTE: use a custom logging function
from helpers.garage_utility import log_performance
# yapf: enable




class NPO(RLAlgorithm):
    """Natural Policy Gradient Optimization implementation from garage
    (https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/npo.py).
    The only difference is that we provide a custom evaluation.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        sampler (garage.sampler.Sampler): Sampler.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        name (str): The name of the algorithm.

    Note:
        sane defaults for entropy configuration:
            - entropy_method='max', center_adv=False, stop_gradient=True
              (center_adv normalizes the advantages tensor, which will
              significantly alleviate the effect of entropy. It is also
              recommended to turn off entropy gradient so that the agent
              will focus on high-entropy actions instead of increasing the
              variance of the distribution.)
            - entropy_method='regularized', stop_gradient=False,
              use_neg_logli_entropy=False

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 sampler,
                 log_every_n_steps=1,
                 scope=None,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 name='NPO'):

        self._log_every_n_steps = log_every_n_steps
        self.policy = policy
        self._scope = scope
        self.max_episode_length = env_spec.max_episode_length
        self._env_spec = env_spec
        self._baseline = baseline
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon
        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._old_policy = policy.clone('old_policy')
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._pg_loss = pg_loss
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = FirstOrderOptimizer

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')

        self._optimizer = make_optimizer(optimizer, **optimizer_args)
        self._lr_clip_range = float(lr_clip_range)
        self._max_kl_step = float(max_kl_step)
        self._policy_ent_coeff = float(policy_ent_coeff)

        self._f_rewards = None
        self._f_returns = None
        self._f_policy_kl = None
        self._f_policy_entropy = None
        self._policy_network = None
        self._old_policy_network = None

        self._episode_reward_mean = collections.deque(maxlen=100)

        self._sampler = sampler

        self._init_opt()

    def _init_opt(self):
        """Initialize optimizater."""
        pol_loss_inputs, pol_opt_inputs = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs

        pol_loss, pol_kl = self._build_policy_loss(pol_loss_inputs)
        self._optimizer.update_opt(loss=pol_loss,
                                   target=self.policy,
                                   leq_constraint=(pol_kl, self._max_kl_step),
                                   inputs=flatten_inputs(
                                       self._policy_opt_inputs),
                                   constraint_name='mean_kl')

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, which rovides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in trainer.step_epochs():

            start_time = time.time()
            trainer.step_episode = trainer.obtain_episodes(trainer.step_itr)
            last_return = self._train_once(trainer.step_itr,
                                           trainer.step_episode,
                                           # FIXME: the inner environment should be public
                                           train_env=trainer.env._env,
                                           val_env=trainer.val_env,
                                           start_time=start_time)
            trainer.step_itr += 1

        return last_return

    def _train_once(self, itr, episodes, train_env=None, val_env=None, start_time=None):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            numpy.float64: Average return.

        """
        # -- Stage: Calculate and pad baselines
        obs = [
            self._baseline.predict({'observations': obs})
            for obs in episodes.observations_list
        ]
        baselines = pad_batch_array(np.concatenate(obs), episodes.lengths,
                                    self.max_episode_length)

        logger.log('Optimizing policy...')
        self._optimize_policy(episodes, baselines)

        # NOTE: keep track of time excluding the evaluation
        if start_time is not None:
            end_time = time.time()
            elapsed = end_time - start_time
            tabular.record('Extras/TrueEpochTime', elapsed)

        # -- Stage: Run and calculate performance of the algorithm
        # NOTE: custom logging function
        undiscounted_returns = log_performance(itr,
                                               episodes,
                                               validate_every_n_steps=self._log_every_n_steps,
                                               train_env=train_env,
                                               val_env=val_env,
                                               policy=self.policy,
                                               discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))

        return np.mean(undiscounted_returns)

    def _optimize_policy(self, episodes, baselines):
        """Optimize policy.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        """
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)
        logger.log('Computing loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)
        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)
        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        ent = np.sum(pol_ent) / np.sum(episodes.lengths)
        tabular.record('{}/Entropy'.format(self.policy.name), ent)
        tabular.record('{}/Perplexity'.format(self.policy.name), np.exp(ent))
        returns = self._fit_baseline_with_data(episodes, baselines)

        ev = explained_variance_1d(baselines, returns, episodes.valids)

        tabular.record('{}/ExplainedVariance'.format(self._baseline.name), ev)
        self._old_policy.parameters = self.policy.parameters

    def _build_inputs(self):
        """Build input variables.

        Returns:
            namedtuple: Collection of variables to compute policy loss.
            namedtuple: Collection of variables to do policy optimization.

        """
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        with tf.name_scope('inputs'):
            obs_var = observation_space.to_tf_placeholder(name='obs',
                                                          batch_dims=2)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, None],
                                                  name='reward')
            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')
            baseline_var = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, None],
                                                    name='baseline')

            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

        augmented_obs_var = obs_var
        for k in self.policy.state_info_keys:
            extra_state_var = policy_state_info_vars[k]
            extra_state_var = tf.cast(extra_state_var, tf.float32)
            augmented_obs_var = tf.concat([augmented_obs_var, extra_state_var],
                                          -1)

        self._policy_network = self.policy.build(augmented_obs_var,
                                                 name='policy')
        self._old_policy_network = self._old_policy.build(augmented_obs_var,
                                                          name='policy')

        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var,
            policy_state_info_vars=policy_state_info_vars,
        )
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
        )

        return policy_loss_inputs, policy_opt_inputs

    # pylint: disable=too-many-branches, too-many-statements
    def _build_policy_loss(self, i):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy loss.
            tf.Tensor: Mean policy KL divergence.

        """
        policy_entropy = self._build_entropy_term(i)
        rewards = i.reward_var

        if self._maximum_entropy:
            with tf.name_scope('augmented_rewards'):
                rewards = i.reward_var + (self._policy_ent_coeff *
                                          policy_entropy)

        with tf.name_scope('policy_loss'):
            adv = compute_advantages(self._discount,
                                     self._gae_lambda,
                                     self.max_episode_length,
                                     i.baseline_var,
                                     rewards,
                                     name='adv')

            adv = tf.reshape(adv, [-1, self.max_episode_length])
            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self._center_adv:
                adv = center_advs(adv, axes=[0], eps=eps)

            if self._positive_adv:
                adv = positive_advs(adv, eps)

            old_policy_dist = self._old_policy_network.dist
            policy_dist = self._policy_network.dist

            with tf.name_scope('kl'):
                kl = old_policy_dist.kl_divergence(policy_dist)
                pol_mean_kl = tf.reduce_mean(kl)

            # Calculate vanilla loss
            with tf.name_scope('vanilla_loss'):
                ll = policy_dist.log_prob(i.action_var, name='log_likelihood')
                vanilla = ll * adv

            # Calculate surrogate loss
            with tf.name_scope('surrogate_loss'):
                lr = tf.exp(ll - old_policy_dist.log_prob(i.action_var))
                surrogate = lr * adv

            # Finalize objective function
            with tf.name_scope('loss'):
                if self._pg_loss == 'vanilla':
                    # VPG uses the vanilla objective
                    obj = tf.identity(vanilla, name='vanilla_obj')
                elif self._pg_loss == 'surrogate':
                    # TRPO uses the standard surrogate objective
                    obj = tf.identity(surrogate, name='surr_obj')
                elif self._pg_loss == 'surrogate_clip':
                    lr_clip = tf.clip_by_value(lr,
                                               1 - self._lr_clip_range,
                                               1 + self._lr_clip_range,
                                               name='lr_clip')
                    surr_clip = lr_clip * adv
                    obj = tf.minimum(surrogate, surr_clip, name='surr_obj')

                if self._entropy_regularzied:
                    obj += self._policy_ent_coeff * policy_entropy

                # filter only the valid values
                obj = tf.boolean_mask(obj, i.valid_var)
                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                loss = -tf.reduce_mean(obj)

            # Diagnostic functions
            self._f_policy_kl = tf.compat.v1.get_default_session(
            ).make_callable(pol_mean_kl,
                            feed_list=flatten_inputs(self._policy_opt_inputs))

            self._f_rewards = tf.compat.v1.get_default_session().make_callable(
                rewards, feed_list=flatten_inputs(self._policy_opt_inputs))

            returns = discounted_returns(self._discount,
                                         self.max_episode_length, rewards)
            self._f_returns = tf.compat.v1.get_default_session().make_callable(
                returns, feed_list=flatten_inputs(self._policy_opt_inputs))

            return loss, pol_mean_kl

    def _build_entropy_term(self, i):
        """Build policy entropy tensor.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy entropy.

        """
        pol_dist = self._policy_network.dist

        with tf.name_scope('policy_entropy'):
            if self._use_neg_logli_entropy:
                policy_entropy = -pol_dist.log_prob(i.action_var,
                                                    name='policy_log_likeli')
            else:
                policy_entropy = pol_dist.entropy()

            # This prevents entropy from becoming negative for small policy std
            if self._use_softplus_entropy:
                policy_entropy = tf.nn.softplus(policy_entropy)

            if self._stop_entropy_gradient:
                policy_entropy = tf.stop_gradient(policy_entropy)

        # dense form, match the shape of advantage
        policy_entropy = tf.reshape(policy_entropy,
                                    [-1, self.max_episode_length])

        self._f_policy_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs), policy_entropy)

        return policy_entropy

    def _fit_baseline_with_data(self, episodes, baselines):
        """Update baselines from samples.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        Returns:
            np.ndarray: Augment returns.

        """
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)

        returns_tensor = self._f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1)

        paths = []
        valids = episodes.valids
        observations = episodes.padded_observations

        # Compute returns
        for ret, val, ob in zip(returns_tensor, valids, observations):
            returns = ret[val.astype(np.bool)]
            obs = ob[val.astype(np.bool)]
            paths.append(dict(observations=obs, returns=returns))

        # Fit baseline
        logger.log('Fitting baseline...')
        self._baseline.fit(paths)
        return returns_tensor

    def _policy_opt_input_values(self, episodes, baselines):
        """Map episode samples to the policy optimizer inputs.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        agent_infos = episodes.padded_agent_infos
        policy_state_info_list = [
            agent_infos[k] for k in self.policy.state_info_keys
        ]

        actions = [
            self._env_spec.action_space.flatten_n(act)
            for act in episodes.actions_list
        ]
        padded_actions = pad_batch_array(np.concatenate(actions),
                                         episodes.lengths,
                                         self.max_episode_length)

        # pylint: disable=unexpected-keyword-arg
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=episodes.padded_observations,
            action_var=padded_actions,
            reward_var=episodes.padded_rewards,
            baseline_var=baselines,
            valid_var=episodes.valids,
            policy_state_info_vars_list=policy_state_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration.

        Args:
            entropy_method (str): A string from: 'max', 'regularized',
                'no_entropy'. The type of entropy method to use. 'max' adds the
                dense entropy to the reward for each time step. 'regularized'
                adds the mean entropy to the surrogate objective. See
                https://arxiv.org/abs/1805.00909 for more details.
            center_adv (bool): Whether to rescale the advantages
                so that they have mean 0 and standard deviation 1.
            stop_entropy_gradient (bool): Whether to stop the entropy gradient.
            use_neg_logli_entropy (bool): Whether to estimate the entropy as
                the negative log likelihood of the action.
            policy_ent_coeff (float): The coefficient of the policy entropy.
                Setting it to zero would mean no entropy regularization.

        Raises:
            ValueError: If center_adv is True when entropy_method is max.
            ValueError: If stop_gradient is False when entropy_method is max.
            ValueError: If policy_ent_coeff is non-zero when there is
                no entropy method.
            ValueError: If entropy_method is not one of 'max', 'regularized',
                'no_entropy'.

        """
        del use_neg_logli_entropy

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
            self._maximum_entropy = True
            self._entropy_regularzied = False
        elif entropy_method == 'regularized':
            self._maximum_entropy = False
            self._entropy_regularzied = True
        elif entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')
            self._maximum_entropy = False
            self._entropy_regularzied = False
        else:
            raise ValueError('Invalid entropy_method')

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_policy_opt_inputs']
        del data['_f_policy_entropy']
        del data['_f_policy_kl']
        del data['_f_rewards']
        del data['_f_returns']
        del data['_policy_network']
        del data['_old_policy_network']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self._name_scope = tf.name_scope(self._name)
        self._init_opt()

########################################################################################################################


class VPG(NPO):
    """Vanilla Policy Gradient implementation from garage
    (https://github.com/rlworkgroup/garage/blob/master/src/garage/tf/algos/vpg.py).
    The only difference is that it relies on the above NPO implementation.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        sampler (garage.sampler.Sampler): Sampler.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        name (str): The name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 sampler,
                 log_every_n_steps=1,
                 scope=None,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 name='VPG'):
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_optimization_epochs=1,
            )
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         sampler=sampler,
                         log_every_n_steps=log_every_n_steps,
                         scope=scope,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         pg_loss='vanilla',
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=use_neg_logli_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         name=name)


class SAC(sac):
    """Soft Actor Critic implementation in pytorch from garage
    (https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/sac.py).

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        sampler (garage.sampler.Sampler): Sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr (float): learning rate for policy optimizers.
        qf_lr (float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        eval_env (Environment): environment used for collecting evaluation
            episodes. If None, a copy of the train env is used.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.
        temporal_regularization_factor (float): coefficient that determines
            the temporal regularization penalty as defined in CAPS as lambda_t
        spatial_regularization_factor (float): coefficient that determines
            the spatial regularization penalty as defined in CAPS as lambda_s
        spatial_regularization_eps (float): sigma of the normal distribution
            from with spatial regularization observations are drawn,
            in caps this is defined as epsilon_s
        per (bool): for activating the prioritized experience replay
    """
    

    def __init__(self,
                 env_spec,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 sampler,
                 *,  # Everything after this is numbers.
                 max_episode_length_eval=None,
                 gradient_steps_per_itr=1,
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 discount=0.99,
                 buffer_batch_size=64,
                 min_buffer_size=1e4,
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 steps_per_epoch=1,
                 num_evaluation_episodes=10,
                 eval_env=None,
                 use_deterministic_evaluation=True,
                 temporal_regularization_factor=0,
                 spatial_regularization_factor=0,
                 spatial_regularization_eps=1,
                 per = False):

        super().__init__(env_spec=env_spec,
                        policy=policy,
                        qf1=qf1,
                        qf2=qf2,
                        replay_buffer=replay_buffer,
                        sampler=sampler,
                        max_episode_length_eval=max_episode_length_eval,
                        gradient_steps_per_itr=gradient_steps_per_itr,
                        fixed_alpha=fixed_alpha,
                        target_entropy=target_entropy,
                        initial_log_entropy=initial_log_entropy,
                        discount=discount,
                        buffer_batch_size=buffer_batch_size,
                        min_buffer_size=min_buffer_size,
                        target_update_tau=target_update_tau,
                        policy_lr=policy_lr,
                        qf_lr=qf_lr,
                        reward_scale=reward_scale,
                        optimizer=optimizer,
                        steps_per_epoch=steps_per_epoch,
                        num_evaluation_episodes=num_evaluation_episodes,
                        eval_env=eval_env,
                        use_deterministic_evaluation=use_deterministic_evaluation,
                        #temporal_regularization_factor=temporal_regularization_factor,
                        #spatial_regularization_factor=spatial_regularization_factor,
                        #spatial_regularization_eps=spatial_regularization_eps
                         )

        self.per = per
        
    def _critic_objective(self, samples_data):
            """Compute the Q-function/critic loss.
            Args:
                samples_data (dict): Transitions(S,A,R,S') that are sampled from
                    the replay buffer. It should have the keys 'observation',
                    'action', 'reward', 'terminal', and 'next_observations'.
            Note:
                samples_data's entries should be torch.Tensor's with the following
                shapes:
                    observation: :math:`(N, O^*)`
                    action: :math:`(N, A^*)`
                    reward: :math:`(N, 1)`
                    terminal: :math:`(N, 1)`
                    next_observation: :math:`(N, O^*)`
            Returns:
                torch.Tensor: loss from 1st q-function after optimization.
                torch.Tensor: loss from 2nd q-function after optimization.
            """
            obs = samples_data['observation']
            actions = samples_data['action']
            rewards = samples_data['reward'].flatten()
            terminals = samples_data['terminal'].flatten()
            next_obs = samples_data['next_observation']
            with torch.no_grad():
                alpha = self._get_log_alpha(samples_data).exp()

            q1_pred = self._qf1(obs, actions)
            q2_pred = self._qf2(obs, actions)

            new_next_actions_dist = self.policy(next_obs)[0]
            new_next_actions_pre_tanh, new_next_actions = (
                new_next_actions_dist.rsample_with_pre_tanh_value())
            new_log_pi = new_next_actions_dist.log_prob(
                value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

            value = alpha * new_log_pi
            target_q_values = torch.min(
                self._target_qf1(next_obs, new_next_actions),
                self._target_qf2(
                    next_obs, new_next_actions)).flatten() - value
            with torch.no_grad():
                q_target = rewards * self._reward_scale + (
                    1. - terminals) * self._discount * target_q_values
            qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
            qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

            return qf1_loss, qf2_loss
    
    
    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from the Policy/Actor.
        """
        obs = samples_data['observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(obs, new_actions),
                                      self._qf2(obs, new_actions))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten())
        if self.per :
            self.replay_buffer.update_priorities(torch.abs(policy_objective))
            policy_objective = torch.mul(policy_objective,  torch.tensor(self.replay_buffer.normalized_weights))
        return policy_objective.mean()
    
class PPO(ppo):
    """Proximal Policy Optimization (PPO).
    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 lr_clip_range=2e-1,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.97,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy'):
        
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         lr_clip_range=lr_clip_range,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)
