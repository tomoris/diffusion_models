import random

import numpy as np
from scipy.stats import multivariate_normal


class NormalDistribution:
    def __init__(self, mu, sigma):
        assert np.array_equal(sigma, sigma.T)
        self._mu = mu
        self._sigma = sigma

        self._inv_sigma = np.linalg.inv(sigma)

    def calc_prob(self, X):
        return multivariate_normal.pdf(X, mean=self._mu, cov=self._sigma)

    def sample(self, N):
        return np.random.multivariate_normal(self._mu, self._sigma, N)

    def calc_score(self, X):
        return -1.0 * (
            np.broadcast_to(
                self._inv_sigma.reshape(1, X.shape[1], X.shape[1]),
                (X.shape[0], X.shape[1], X.shape[1]),
            )
            @ (X - self._mu).reshape(X.shape[0], X.shape[1], 1)
        ).reshape(X.shape[0], X.shape[1])


class ContaminatedNormalDistribution:
    def __init__(self, dist_params_list):
        sum_ratio = sum([ratio for _, _, ratio in dist_params_list])
        epsilon = 0.0001
        assert 1.0 - epsilon < sum_ratio < 1.0 + epsilon
        mu0, sigma0, _ = dist_params_list[0]
        self._dist_list = []
        self._ratio_list = []
        for mu, sigma, ratio in dist_params_list:
            assert mu0.shape == mu.shape
            assert sigma0.shape == sigma.shape
            dist = NormalDistribution(mu, sigma)
            self._dist_list.append(dist)
            self._ratio_list.append(ratio)

    def calc_prob(self, X):
        prob = np.zeros(X.shape[0])
        prob_list = []
        for i, dist in enumerate(self._dist_list):
            ratio = self._ratio_list[i]
            tmp_prob = dist.calc_prob(X)
            prob += ratio * tmp_prob
            prob_list.append(tmp_prob)
        return prob, prob_list

    def sample(self, N):
        dist_indexes = np.random.choice(
            len(self._ratio_list), N, p=[ratio for ratio in self._ratio_list]
        )
        tmp_N_list = [
            np.count_nonzero(dist_indexes == i) for i in range(len(self._ratio_list))
        ]
        samples = []
        for i, tmp_N in enumerate(tmp_N_list):
            samples.append(self._dist_list[i].sample(tmp_N))
        random.shuffle(samples)
        samples = np.concatenate(samples, axis=0)
        return samples

    def calc_score(self, X):
        epsilon = np.finfo(np.float32).tiny
        score = np.zeros((X.shape[0], X.shape[1]))
        prob, prob_list = self.calc_prob(X)
        for i, dist in enumerate(self._dist_list):
            ratio = self._ratio_list[i]
            score += (
                ratio
                * dist.calc_score(X)
                * (
                    np.broadcast_to(
                        prob_list[i].reshape(X.shape[0], 1), (X.shape[0], X.shape[1])
                    )
                    + epsilon
                )
            )
        score /= (
            np.broadcast_to(prob.reshape(X.shape[0], 1), (X.shape[0], X.shape[1]))
            + epsilon
        )
        return score


def main():
    dist_params_list = []

    mu = np.array([-10.0, 10.0])
    sigma = np.array([[0.5, 0], [0, 0.5]])
    ratio = 1 / 6
    dist_params_list.append((mu, sigma, ratio))

    mu = np.array([0.0, -7.5])
    sigma = np.array([[0.5, 0], [0, 0.5]])
    ratio = 1 / 6
    dist_params_list.append((mu, sigma, ratio))

    mu = np.array([10.0, 10.0])
    sigma = np.array([[0.5, 0], [0, 0.5]])
    ratio = 1 / 6
    dist_params_list.append((mu, sigma, ratio))

    mu = np.array([0.0, 2.5])
    sigma = np.array([[1.25, 0], [0, 1.25]])
    ratio = 1 / 6
    dist_params_list.append((mu, sigma, ratio))

    mu = np.array([-6.0, -2.5])
    sigma = np.array([[2, 1.5], [1.5, 2]])
    ratio = 1 / 6
    dist_params_list.append((mu, sigma, ratio))

    mu = np.array([2.5, -2.5])
    sigma = np.array([[2, 1.5], [1.5, 2]])
    ratio = 1 / 6
    dist_params_list.append((mu, sigma, ratio))

    p = ContaminatedNormalDistribution(dist_params_list)
    sampled_X = p.sample(10000)

    np.savetxt("data/sampled_X.numpy.txt", sampled_X)


if __name__ == "__main__":
    main()
