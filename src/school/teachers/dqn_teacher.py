class DQNTeacher(object):
    """Deep Q-learning agent."""

    def __init__(self):
        """Set parameters, initialize network."""
        pass

    def step(self, observation, training=True):

        pass

    def policy(self, state, training):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        pass

    def update_target_network(self):
        """Update target network weights with current online network values."""
        pass

    def train_network(self):
        """Update online network weights."""
        pass