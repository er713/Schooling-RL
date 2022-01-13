import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.metrics import categorical_crossentropy


def actor_loss(logits, action, δ):
    action_probabilities = tfp.distributions.Categorical(logits=logits)
    log_prob = action_probabilities.log_prob(action)

    return -log_prob * δ


def actor_loss2(logits, action, δ):
    mask = -11. * tf.one_hot(action, logits.shape[0], dtype=tf.float32) + tf.ones_like(logits)
    loss = tf.multiply(logits, mask / 10.)

    return δ * tf.reduce_sum(loss)


def actor_loss3(logits, action, δ):
    loss = categorical_crossentropy(tf.one_hot(action, logits.shape[1]), logits[0], from_logits=True)
    return δ*loss


def actor_batch_loss(logits, actions, d):
    r = []
    for l, a, _d in zip(logits, actions, d):
        print('!!!!!!!!!!!!!!!!! l', l, l.shape)
        print('!!!!!!!!!!!!!!!!!a', a, a.shape)
        print('!!!!!!!!!!!!!!!!_d', _d, _d.shape)
        r.append(actor_loss(l, a, _d))
    return sum(r) / float(len(r))
