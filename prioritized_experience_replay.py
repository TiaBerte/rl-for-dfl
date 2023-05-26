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

