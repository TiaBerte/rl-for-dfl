import random 
from garage.replay_buffer.replay_buffer import ReplayBuffer as replayBuffer


class ReplayBuffer(replayBuffer):
    """Replay Buffer implementation from garage
    (https://github.com/rlworkgroup/garage/blob/master/src/garage/replay_buffer/replay_buffer.py)

    Method sample_transition need to be implemented. In GitHub it's an abstrac method
    """

    def __init__(self, env_spec, size_in_transitions, time_horizon):
        super.__init__(env_spec, size_in_transitions, time_horizon)


    def sample(self, batch_size):
        """Sample a transition of batch_size.
        Args:
            batch_size(int): The number of transitions to be sampled.
        """
        indexes = random .sample(range(0, self._n_transitions_stored), batch_size)
        samples = {}

        for key in self._buffer:
            samples[key] =  [self._buffer[key][id] for id in indexes]

        return samples



