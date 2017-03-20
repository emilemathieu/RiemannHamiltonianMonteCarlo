import numpy as np
import timeit
from tools import LogNormPDF


def log_multi_normal_density(X, means, cov):
    dim = X.shape[1]
    det_cov = np.linalg.det(cov)
    u = X-means
    return 1/(np.sqrt((2*np.pi)**dim)*det_cov) * np.exp(-0.5*u.T.dot(np.linalg.inv(cov).dot(u)))


def iwls(XX, t, alpha=100, max_iter=10000, burn_in=5000):
    print("--- Initialization...")
    # Initialization
    accepted = 0
    n_samples, dim = XX.shape
    beta = np.zeros(dim)
    beta_saved = np.zeros((max_iter-burn_in, dim))

    # Computing joint likelihood for current beta
    log_prior = LogNormPDF(np.zeros((1, dim)), beta[:, np.newaxis], alpha)
    f = np.dot(XX, beta)
    log_likelihood = np.dot(f.T, t) - np.sum(np.log(1+np.exp(f)))
    current_LJL = log_likelihood + log_prior

    # Computing weights
    p = 1 / (1 + np.exp(-np.sum(beta*XX, axis=1)))
    W = p * (np.ones(n_samples) - p)
    inv_W = np.eye(W.shape[0]) / W

    # Computing current covariance and means
    current_cov = np.linalg.inv(np.eye(dim) / alpha + (XX.T*np.tile(W[:, np.newaxis].T, (dim, 1))).dot(XX))  # covariance matrix
    z = XX.dot(beta)[:, np.newaxis] + inv_W.dot(t - p[:, np.newaxis])
    current_mean = current_cov.dot(XX.T.dot(np.diag(W).dot(z)))[:, 0]  # means vector

    print("--- Iterating...")
    for i in range(max_iter):
        if i % 1000 == 0:
            print("Iteration %d" % i)
        if i == burn_in:
            print("Burn-in complete, now drawing posterior samples.")
            start = timeit.default_timer()
        # Sampling from the proposal density
        beta_new = np.random.multivariate_normal(current_mean, current_cov)

        # Computing joint likelihood for this new beta
        log_prior = LogNormPDF(np.zeros((1, dim)), beta_new[:, np.newaxis], alpha)
        f = np.dot(XX, beta_new)
        log_likelihood = np.dot(f.T, t) - np.sum(np.log(1 + np.exp(f)))
        proposed_LJL = log_likelihood + log_prior

        # Computing new weights
        p = 1 / (1 + np.exp(-np.sum(beta_new*XX, axis=1)))
        W = p * (np.ones(n_samples) - p)
        inv_W = np.eye(W.shape[0]) / W

        # Computing new covariance and means
        new_cov = np.linalg.inv(np.eye(dim) / alpha + (XX.T*np.tile(W[:, np.newaxis].T, (dim, 1))).dot(XX))  # new covariance matrix
        z = XX.dot(beta_new)[:, np.newaxis] + inv_W.dot(t - p[:, np.newaxis])
        new_mean = new_cov.dot(XX.T.dot(np.diag(W).dot(z)))[:, 0]  # new means vector

        # Computing proposal probabilities (without the normalizing constants)
        prob_new_given_old = -np.sum(np.diag(np.log(np.linalg.cholesky(current_cov+np.eye(dim)*1e-6))))
        prob_new_given_old -= 0.5 * (beta_new - current_mean).T.dot(np.linalg.inv(
            current_cov)).dot(beta_new - current_mean)

        prob_old_given_new = -np.sum(np.diag(np.log(np.linalg.cholesky(new_cov+np.eye(dim)*1e-6))))
        prob_old_given_new -= 0.5 * (beta - new_mean).T.dot(np.linalg.inv(
            new_cov)).dot(beta - new_mean)

        # Acceptance ratio
        ratio = proposed_LJL + prob_old_given_new - current_LJL - prob_new_given_old

        # Parameters update if the new sample is accepted
        if ratio > 0 or ratio > np.log(np.random.uniform()):
            accepted += 1
            beta = beta_new
            current_cov = new_cov
            current_mean = new_mean
            current_LJL = proposed_LJL

        # Saving the posterior betas
        if i >= burn_in:
            beta_saved[i - burn_in] = beta
    print("--- Iterating: done.")
    print("Number of accepted samples: ", accepted)
    time = timeit.default_timer() - start
    return beta_saved, time
