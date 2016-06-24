import tensorflow as tf
import numpy as np
import prettytensor as pt


def construct_policy_net(obs, action_dim):
    mean = (
        pt.wrap(obs).fully_connected(64, activation_fn=tf.nn.relu, stddev=0.01).
        fully_connected(64, activation_fn=tf.nn.relu, stddev=0.01).
        fully_connected(action_dim, activation_fn=None, stddev=0.01))
#     input_dim = obs.get_shape()[1]
    # zero_weight = tf.get_variable('zero_weight', shape=[input_dim, action_dim],
                                  # initializer=tf.constant_initializer(0.0))
    # zero_input = tf.matmul(obs, zero_weight)
    # logstd = tf.get_variable('logstd', shape=[1, action_dim],
                             # initializer=tf.truncated_normal_initializer(stddev=0.1))
    # std = tf.exp(logstd)
    # std = zero_input + std
    action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, action_dim)).astype(np.float32))
    action_dist_logstd = tf.tile(action_dist_logstd_param, tf.pack((tf.shape(mean)[0], 1)))
    std = tf.exp(action_dist_logstd)
    return tf.concat(1, [mean, std])
