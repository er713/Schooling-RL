import tensorflow_probability as tfp


def actor_loss(logits, action, δ):
    action_probabilities = tfp.distributions.Categorical(logits=logits)
    log_prob = action_probabilities.log_prob(action)

    return -log_prob * δ
