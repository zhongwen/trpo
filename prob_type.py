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
        return np.random.rand(mean_nd.shape[0], self.d) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]


def test_diag_gauss():
    N = 100000
    dim = 3
    mean0 = tf.placeholder(tf.float32, shape=[None, dim], name="mean0")
    std0 = tf.placeholder(tf.float32, shape=[None, dim], name="std0")
    mean1 = tf.placeholder(tf.float32, shape=[None, dim], name="mean1")
    std1 = tf.placeholder(tf.float32, shape=[None, dim], name="std1")
    prob0 = MeanStd(mean0, std0)
    prob1 = MeanStd(mean1, std1)
    gauss = DiagGauss(dim)
    ent0 = gauss.entropy(prob0)
    kl = gauss.kl(prob0, prob1)
    x = gauss.sample(prob0)
    loglikelihood = gauss.loglikelihood(x, prob0)
    loglikelihood2 = gauss.loglikelihood(x, prob1)
    with tf.Session() as sess:
        mean0_ = np.random.rand(1, dim)
        std0_ = np.random.rand(1, dim)
        mean1_ = np.random.rand(1, dim)
        std1_ = np.random.rand(1, dim)
        mean0_np = np.tile(mean0_, [N, 1])
        std0_np = np.tile(std0_, [N, 1])
        mean1_np = np.tile(mean1_, [N, 1])
        std1_np = np.tile(std1_, [N, 1])
        kl_val, ent0_val, loglik_val, loglik_val2 = sess.run(
            [kl, ent0, loglikelihood, loglikelihood2],
            {prob0.mean: mean0_np, prob0.std: std0_np, prob1.mean: mean1_np,
             prob1.std: std1_np})
        entval_ll = -np.mean(loglik_val)
        entval_ll_std = np.std(loglik_val) / np.sqrt(N)
        assert np.abs(entval_ll - np.mean(ent0_val)) < 3 * entval_ll_std

        kl = np.mean(kl_val)
        kl_ll = - np.mean(ent0_val) - np.mean(loglik_val2)
        kl_ll_std = np.std(loglik_val2) / np.sqrt(N)
        assert np.abs(kl - kl_ll) < 3 * kl_ll_std

if __name__ == '__main__':
    test_diag_gauss()
