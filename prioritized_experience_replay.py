import numpy as np
from garage.replay_buffer.path_buffer import PathBuffer
from dowel import tabular
from garage.np import discount_cumsum
from garage import StepType


class PrioritizedReplay(PathBuffer):
    """Custom implementation of Prioritized Experience Replay from
    github.com/davidrpugh/stochastic-expatriate-descent/blob/master/_notebooks/2020-04-14-prioritized-experience-replay.ipynb
    which extends the PathBuffer class from garage.
    Args:
        
    """
    def __init__(self,  capacity_in_transitions, env_spec=None, alpha=0.6, 
                 beta_start = 0.4, annealing_rate = 3e-3):
        super().__init__(capacity_in_transitions, env_spec)
        self._alpha = alpha
        self._n_episodes = 0
        self._beta_start = beta_start
        self._annealing_rate = annealing_rate
        self._priorities_buffer = np.ones(self._capacity)
       

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self.n_transitions_stored == 0
    
    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self.n_transitions_stored == self._capacity
    
    def sample_transitions(self, batch_size):
        """Sample a batch of experiences from memory."""
        beta = self.beta_annealing_schedule()
        ps = self._priorities_buffer[:self.n_transitions_stored] + 1e-8
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        self.idxs = np.random.choice(np.arange(ps.size),
                                         size=batch_size,
                                         replace=False,
                                         p=sampling_probs).astype('int32')
        experiences = {key: buf_arr[self.idxs] for key, buf_arr in self._buffer.items()}
        weights = np.zeros(len(self.idxs))
        for i in range(len(self.idxs)):
            weights[i] = (self.n_transitions_stored * sampling_probs[self.idxs[i]])**-beta
        #weights = (self.n_transitions_stored * sampling_probs[self.idxs])**-beta
        self.normalized_weights = weights / weights.max()
        
        return experiences #, normalized_weights

    
    def update_priorities(self, priorities):
        """Update the priorities of sampled experiences."""
        self._priorities_buffer[self.idxs] = priorities.cpu().detach().numpy()
        idx_ones = np.where(np.array(self._priorities_buffer) == 1)[0]
        self._priorities_buffer[idx_ones] = np.mean(priorities.cpu().detach().numpy())


    def beta_annealing_schedule(self):
        self._n_episodes += 1
        return 1 - (1 - self._beta_start)*np.exp(-self._annealing_rate * self._n_episodes)

'''
class EmphasizingRecentExperience(PrioritizedReplay):
     def __init__(self,  capacity_in_transitions, env_spec=None, alpha=0.6, 
                  beta_start = 0.4, annealing_rate = 6e-3, max_epochs=2000, eta_0=0.996, eta_T=1.0, c_k_min=2500):
        super().__init__(capacity_in_transitions, env_spec, alpha, beta_start, annealing_rate)
        self.max_epochs = max_epochs
        self.eta_0 = eta_0
        self.eta_T = eta_T
        self.c_k_min = c_k_min

     def sample_transitions(self, batch_size):
        """Sample a batch of experiences from memory."""
        print('SAMPLE TRANSITION with ERE')
        print('transition stored', self.n_transitions_stored)
        eta_t = self.eta_0 + (self.eta_T - self.eta_0)*(self._n_episodes/self.max_epochs)
        c_k = max(int(self.n_transitions_stored*eta_t), self.c_k_min)
        c_k = min(c_k, self.n_transitions_stored)
        beta = self.beta_annealing_schedule()
        # use sampling scheme to determine which experiences to use for learning
        ps = self._priorities_buffer[:c_k]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)# if self.is_full() else [1/c_k] * c_k
        #print('ps', ps)
        #print('ps size', ps.size)
        #print('samp size', len(sampling_probs))
        #print('sampling prob', sampling_probs)
        self.idxs = np.random.choice(np.arange(ps.size),
                                         size=batch_size,
                                         replace=False,
                                         p=sampling_probs).astype('int32')
        #print(self.idxs)
        #print(type(self.idxs))
        # select the experiences and compute sampling weights
        experiences = {key: buf_arr[self.idxs] for key, buf_arr in self._buffer.items()}
        weights = (self.n_transitions_stored * sampling_probs[self.idxs])**-beta
        self.normalized_weights = weights / weights.max()
        
        return experiences #, normalized_weights
'''

def sac_log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []

    batch_feasibles = list()
    batch_true_cost = list()
    batch_regret = list()

    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))

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