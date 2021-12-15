import tensorflow as tf
from .. import losses


@tf.function
def _learn_main(actor: tf.keras.Model, critic: tf.keras.Model, state: tf.Tensor, action: tf.Tensor,
                next_state: tf.Tensor,
                reward: tf.Tensor, done: tf.Tensor, gamma: tf.Tensor, actor_opt, critic_opt) -> None:
    """
    Dokumentacja
    """
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        q = critic(state)
        q_next = critic(next_state)
        logits = actor(state)

        δ = reward + gamma * q_next * (1 - done) - q  # this works w/o tf.function
        # δ = float(reward) + float(gamma * q_next * (1 - done)) - float(q)  # float only for tf.function

        actor_loss = losses.actor_loss(logits, action, δ)
        critic_loss = δ ** 2  # MSE?

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)

    actor_opt.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))
