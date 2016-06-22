import tensorflow as tf
import prettytensor as pt


def construct_policy_net(obs, action_dim, batch_size):
    mean = (
        pt.wrap(obs).fully_connected(64, activation_fn=tf.nn.tanh).
        fully_connected(action_dim, activation_fn=None))
    logstd = tf.get_variable('logstd', shape=[1, action_dim])
    std = tf.exp(logstd)
    std = tf.tile(std, [batch_size, 1], "std")
    return tf.concat(1, [mean, std])
