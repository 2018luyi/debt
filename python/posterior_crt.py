'''Conditional randomization test (CRT) implementation for whether a variable
has any effect on the posterior probability of coming from the alternative.'''
from __future__ import print_function
import sys
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from utils import ilogit, calc_fdr
from normix import one_sided_empirical_null, estimate_mixture, GridDistribution1D

def factorize(X, nfactors=30):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=nfactors)
    W = nmf.fit_transform(X)
    return W, nmf.components_.T

class PosteriorConditionalRandomizationTester:
    def __init__(self, prior, fdr=0.1):
        self.prior = prior
        self.fdr = fdr

    def run(self, nfactors=30, num_sweeps=50, num_alt_bins=220, verbose=False):
        # Fit a factor model for the covariates
        if verbose:
            print('Fitting factor model for features')
        W, V = factorize(self.prior.X, nfactors=nfactors)
        X_probs = W.dot(V.T).clip(1e-4,1-1e-4) # Get the null probabilities

        # Calculate test statistics
        X = np.copy(self.prior.X)
        posteriors = np.zeros((X.shape[0], 2))
        self.tstats = np.zeros(X.shape[1])
        for feat_idx in range(self.prior.X.shape[1]):
            if verbose and feat_idx % 100 == 0:
                print('Feature {}'.format(feat_idx))
            # Get the value of the posteriors with the real feature
            posteriors[np.arange(X.shape[0]), X[:,feat_idx]] = self.prior.posteriors

            # Get the value of the posteriors with the complement feature value
            X[:,feat_idx] = 1-X[:,feat_idx]
            for fold_idx, test_indices in enumerate(self.prior.folds):
                # Use the prior to calculate the null posteriors
                posteriors[test_indices, X[test_indices,feat_idx]] = self.prior.predict(X[test_indices], y=self.prior.y[test_indices], models=[fold_idx])[1]

            # Reset the value to the truth
            X[:,feat_idx] = 1-X[:,feat_idx]

            # Get the test statistics
            all_tstats = posteriors*np.log(posteriors) + (1-posteriors)*np.log(1-posteriors)

            # Sample the null features
            self.tstats[feat_idx] = all_tstats[np.arange(all_tstats.shape[0]), (np.random.random(size=X.shape[0]) <= X_probs[:,feat_idx]).astype(int)].mean()

        # Standardize tstats
        self.tstats_mean = self.tstats.mean()
        self.tstats_std = self.tstats.std()
        self.tstats = (self.tstats - self.tstats_mean) / self.tstats_std

        # Estimate the empirical null and alternative
        self.null_dist = one_sided_empirical_null(self.tstats)
        self.mix = estimate_mixture(self.tstats, self.null_dist)
        self.pi0 = self.mix['w'][0]
        self.alt_dist = self.mix['dist']


    def predictions(self):
        # Get the posterior estimates
        P0 = self.null_dist.pdf(self.tstats)
        P1 = self.alt_dist.pdf(self.tstats)
        post0 = P0 * self.pi0
        post1 = P1 * (1-self.pi0)
        posteriors = (post1 / (post0 + post1))
        self.posteriors = posteriors.clip(1e-8, 1-1e-8)
        self.posteriors[self.tstats > 0] = 0 # prevent taking positive features
        self.discoveries = calc_fdr(self.posteriors, self.fdr)

        return {'tstats': self.tstats,
                'posteriors': self.posteriors,
                'discoveries': self.discoveries}












