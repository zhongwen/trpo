import tensorflow as tf
import prettytensor as pt


def construct_policy_net(obs, action_dim):
    mean = (
        pt.wrap(obs).fully_connected(64, activation_fn=tf.nn.tanh).
        fully_connected(64, activation_fn=tf.nn.tanh).
        fully_connected(action_dim, activation_fn=None))
    input_dim = obs.get_shape()[1]
    zero_weight = tf.get_variable('zero_weight', shape=[input_dim, action_dim],
                                  initializer=tf.constant_initializer(0.0))
    zero_input = tf.matmul(obs, zero_weight)
    logstd = tf.get_variable('logstd', shape=[1, action_dim])
    std = tf.exp(logstd)
    std = zero_input + std
    return tf.concat(1, [mean, std])
