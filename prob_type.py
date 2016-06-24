from __future__ import print_function
import tensorflow as tf
import numpy as np


class ProbType(object):
    # def sampled_variable(self):
        # raise NotImplementedError
    # def prob_variable(self):
        # raise NotImplementedError

    def likelihood(self, a, prob):
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        raise NotImplementedError

    def kl(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class DiagGauss(ProbType):

    def __init__(self, d):
        self.d = d
    # def sampled_variable(self):
        # return T.matrix('a')
    # def prob_variable(self):
        # return T.matrix('prob')

    def loglikelihood(self, a, prob):
        mean0 = tf.slice(prob, [0, 0], [-1, self.d])
        std0 = tf.slice(prob, [0, self.d], [-1, self.d])
        # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
        return - 0.5 * tf.reduce_sum(tf.square((a - mean0) / std0), 1) - 0.5 * tf.log(
            2.0 * np.pi) * self.d - tf.reduce_sum(tf.log(std0), 1)

    def likelihood(self, a, prob):
        return tf.exp(self.loglikelihood(a, prob))

    def kl(self, prob0, prob1):
        mean0 = tf.slice(prob0, [0, 0], [-1, self.d])
        std0 = tf.slice(prob0, [0, self.d], [-1, self.d])
        mean1 = tf.slice(prob1, [0, 0], [-1, self.d])
        std1 = tf.slice(prob1, [0, self.d], [-1, self.d])
        return tf.reduce_sum(
            tf.log(std1 / std0),
            1) + tf.reduce_sum(
            (tf.square(std0) + tf.square(mean0 - mean1)) /
            (2.0 * tf.square(std1)),
            1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = tf.slice(prob, [0, self.d], [-1, self.d])
        return tf.reduce_sum(tf.log(std_nd),
                             1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(mean_nd.shape[0], self.d) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]
