
"""A wrapper that scales and shifts observations."""
import gym


class NormalizeStateWrapper(gym.ObservationWrapper):
    """Wraps an environment to shift and scale observations.
    """

    def __init__(self, env, shift, scale):
        super(NormalizeStateWrapper, self).__init__(env)
        self.shift = shift
        self.scale = scale

    def observation(self, observation):
        return (observation + self.shift) * self.scale

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps  # pylint: disable=protected-access
