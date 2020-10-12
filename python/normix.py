import numpy as np
from scipy.stats import norm as norm
from scipy.optimize import fmin_bfgs
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator

class GridDistribution:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def pdf(self, data):
        # Find the closest bins
        rhs = np.searchsorted(self.x, data)
        lhs = (rhs - 1).clip(0)
        rhs = rhs.clip(0, len(self.x) - 1)

        # Linear approximation (trapezoid rule)
        rhs_dist = np.abs(self.x[rhs] - data)
        lhs_dist = np.abs(self.x[lhs] - data)
        denom = rhs_dist + lhs_dist
        denom[denom == 0] = 1. # handle the zero-distance edge-case
        rhs_weight = 1.0 - rhs_dist / denom
        lhs_weight = 1.0 - rhs_weight

        return lhs_weight * self.y[lhs] + rhs_weight * self.y[rhs]

def trapezoid(x, y):
    return np.sum((x[1:] - x[0:-1]) * (y[1:] + y[0:-1]) / 2.)


class GridDistribution1D:
    def __init__(self, bins, w):
        self.w = w
        self.w /= np.trapz(w, bins)
        self.bins = bins
        self.grid = RegularGridInterpolator((bins,), w, bounds_error=False, fill_value=1e-10)
        
        a = np.concatenate([[0], np.cumsum(((self.bins[1:] - self.bins[:-1]) * (self.w[1:] + self.w[:-1])) / 2).clip(1e-8,1-1e-8)])
        
        too_small = np.abs(a[1:-1] - a[:-2]) <= 1e-3
        a = np.concatenate([[0], a[1:-1][~too_small], [1]])
        b = np.concatenate([[self.bins[0]],self.bins[1:-1][~too_small], [self.bins[-1]]])

        self.ppf_grid = RegularGridInterpolator((a,), b, bounds_error=True)

        # Quick and dirty expectation (TODO: better estimate this)
        self.expectation = (self.grid(self.bins) * self.bins).sum()

    def pdf(self, x):
        return self.grid(x)
        
    def ppf(self, a):
        if np.isscalar(a):
            return self.ppf_grid(np.array([a]))[0]
        return self.ppf_grid(a)

def generate_sweeps(num_sweeps, num_samples):
    results = []
    for sweep in range(num_sweeps):
        a = np.arange(num_samples)
        np.random.shuffle(a)
        results.extend(a)
    return np.array(results)

def predictive_recursion(z, num_sweeps, grid_x, mu0=0., sig0=1.,
                            nullprob=1.0, decay=-0.67):
    sweeporder = generate_sweeps(num_sweeps, len(z))
    theta_guess = np.ones(len(grid_x)) / float(len(grid_x))
    return predictive_recursion_fdr(z, sweeporder, grid_x, theta_guess,
                                    mu0, sig0, nullprob, decay)

def predictive_recursion_fdr(z, sweeporder, grid_x, theta_guess, mu0 = 0.,
                            sig0 = 1.0, nullprob = 1.0, decay = -0.67):
    gridsize = grid_x.shape[0]
    theta_subdens = deepcopy(theta_guess)
    pi0 = nullprob
    joint1 = np.zeros(gridsize)
    ftheta1 = np.zeros(gridsize)

    # Begin sweep through the data
    for i, k in enumerate(sweeporder):
        cc = (3. + i)**decay
        joint1 = norm.pdf(grid_x, loc=z[k] - mu0, scale=sig0) * theta_subdens
        m0 = pi0 * norm.pdf(z[k] - mu0, 0., sig0)
        m1 = trapezoid(grid_x, joint1)
        mmix = m0 + m1
        pi0 = (1. - cc) * pi0 + cc * (m0 / mmix)
        ftheta1 = joint1 / mmix
        theta_subdens = (1. - cc) * theta_subdens + cc * ftheta1

    # Now calculate marginal distribution along the grid points
    y_mix = np.zeros(gridsize)
    y_signal = np.zeros(gridsize)
    for i, x in enumerate(grid_x):
        joint1 = norm.pdf(grid_x, x - mu0, sig0) * theta_subdens
        m0 = pi0 * norm.pdf(x, mu0, sig0)
        m1 = trapezoid(grid_x, joint1)
        y_mix[i] = m0 + m1;
        y_signal[i] = m1 / (1. - pi0)

    return {'grid_x': grid_x,
            'sweeporder': sweeporder,
            'theta_subdens': theta_subdens,
            'pi0': pi0,
            'y_mix': y_mix,
            'y_signal': y_signal}

def empirical_null(z, nmids=150, pct=-0.01, pct0=0.25, df=4, verbose=0):
    '''Estimate f(z) and f_0(z) using a polynomial approximation to Efron (2004)'s method.'''
    N = len(z)
    med = np.median(z)
    lb = med + (1 - pct) * (z.min() - med)
    ub = med + (1 - pct) * (z.max() - med)

    breaks = np.linspace(lb, ub, nmids+1)
    zcounts = np.histogram(z, bins=breaks)[0]
    mids = (breaks[:-1] + breaks[1:])/2

    ### Truncated Polynomial

    # Truncate to [-3, 3]
    selected = np.logical_and(mids >= -2, mids <= 2)
    zcounts = zcounts[selected]
    mids = mids[selected]

    # Form a polynomial basis and multiply by z-counts
    X = np.array([mids ** i for i in range(df+1)]).T
    beta0 = np.zeros(df+1)
    loglambda_loss = lambda beta, X, y: -((X * y[:,np.newaxis]).dot(beta) - np.exp(X.dot(beta).clip(-20,20))).sum() + 1e-6*np.sqrt((beta ** 2).sum())
    results = fmin_bfgs(loglambda_loss, beta0, args=(X, zcounts), disp=verbose)
    a = np.linspace(-3,3,1000)
    B = np.array([a ** i for i in range(df+1)]).T
    beta_hat = results

    # Back out the mean and variance from the Taylor terms
    x_max = mids[np.argmax(X.dot(beta_hat))]
    loglambda_deriv1_atmode = np.array([i * beta_hat[i] * x_max**(i-1) for i in range(1,df+1)]).sum()
    loglambda_deriv2_atmode = np.array([i * (i-1) * beta_hat[i] * x_max**(i-2) for i in range(2,df+1)]).sum()
    
    # Handle the edge cases that arise with numerical precision issues
    sigma_enull = np.sqrt(-1.0/loglambda_deriv2_atmode) if loglambda_deriv2_atmode < 0 else 1
    mu_enull = (x_max - loglambda_deriv1_atmode/loglambda_deriv2_atmode) if loglambda_deriv2_atmode != 0 else 0

    return (mu_enull, sigma_enull)


def one_sided_empirical_null(z, nbins=200):
    '''Histogram poisson regression for estimating the null distribution
    under the assumption that the right-hand side is entirely null and the
    distribution is symmetric.'''
    results = estimate_density(z, bins=nbins)
    bins = results['bins']
    rates = results['z']
    central_peak = np.argmax(rates)

    # Assume the null distribution is symmetric about the central peak
    # and that all values to the right of the central peak are null.
    null_grid = np.concatenate([(2*bins[central_peak] - bins[central_peak+1:])[::-1], bins[central_peak:]])
    null_probs = np.concatenate([rates[central_peak+1:][::-1], rates[central_peak:]])
    null_probs /= null_probs.sum() # normalize to make a proper distribution
    return GridDistribution1D(null_grid, null_probs)

def generate_sweeps(num_sweeps, num_samples):
    results = []
    for sweep in range(num_sweeps):
        a = np.arange(num_samples)
        np.random.shuffle(a)
        results.extend(a)
    return np.array(results)

def generate_bins(z, bins):
    if np.isscalar(bins):
        # Linear grid over the range of z, with 10% overhang on each side
        z_range = z.max() - z.min()
        bins = np.linspace(z.min() - z_range*0.1, z.max() + z_range*0.1, bins)
    return bins


def estimate_density(y, bins=200, weights=None, nsweeps=10, sweeporder=None,
                        decay=-0.67, sigma=None, sbins=30, kernel=None, **kwargs):
    '''Estimates a marginal density as a mixture of normals using predictive
    recursion.'''
    if sweeporder is None:
        # Create random sweeps through the dataset
        sweeporder = generate_sweeps(nsweeps, len(y))

    bins = generate_bins(y, bins)

    if sigma is None:
        if np.isscalar(sbins):
            sbins = np.linspace(1e-2, 4, sbins)
        
        # Estimate the density using a range of kernel bandwidths
        results = [estimate_density(y, sigma=s, sweeporder=sweeporder, bins=bins, **kwargs) for s in sbins]

        # Calculate the 'predictive recursion marginal likelihood' (PRML) of y for each bandwidth
        marginals = [r['logm'] for r in results]

        # Weight each density proportional to the PRML of y
        Zs = [r['z'] for r in results]
        z_logits = np.exp(marginals - np.max(marginals))
        z_probs = z_logits / z_logits.sum()
        z_hat = (z_probs[:,None] * Zs).sum(axis=0)

        # Create the new weighted distribution
        result =  results[np.argmax(marginals)]
        result['dist'] = GridDistribution1D(bins, z_hat)
        result['z'] = z_hat
        result['logm_grid'] = marginals
        result['w'] = (z_probs[:,None] * [r['w'] for r in results]).sum(axis=0)

        return result

    if weights is None:
        # Assign equal weight to every data point
        weights = np.ones_like(y)

    if kernel is None or kernel == 'normal':
        kernel = lambda x, b, s: norm.pdf(x[:,None], b[None], scale=s)

    # Initialize everything to equal weights
    w = np.ones(len(bins)) / (bins.max() - bins.min())

    # Calculate the likelihood of each y coming from a N(mu, 1) centered at this bin
    likelihoods = kernel(y, bins, sigma)

    # Sweep through the data and reweight the density using each point iteratively
    log_marginal = 0
    for i, k in enumerate(sweeporder):
        step_weight = (3. + i)**decay # Each iteration contributes slightly less
        f = w * likelihoods[k] # prob of z_k coming from N(bins, 1) * current prior
        m = np.trapz(f, bins)
        if i < len(y):
            log_marginal += np.log(max(1e-10, m))
        w = (1. - step_weight * weights[k]) * w + step_weight * weights[k] * f/m # reweight

    # Get the marginal likelihood of each point to create a grid approximation
    z = (w[None]*kernel(bins, bins, sigma)).sum(axis=1)
    z /= np.trapz(z, bins)

    return {'bins': bins, 'dist': GridDistribution1D(bins, z), 'w': w, 'z': z, 'logm': log_marginal, 'sweeporder': sweeporder}


def estimate_mixture(y, f0, bins=200, weights=None, nsweeps=10, sweeporder=None,
                            decay=-0.67, sigma=None, sbins=30, kernel=None, **kwargs):
    '''Estimates a 2-component mixture density, w*f0 + (1-w)*f1, where f0 is
    assumed to be known. Uses predictive recursion to estimate f1 and w, then
    returns f1.'''
    if sweeporder is None:
        # Create random sweeps through the dataset
        sweeporder = generate_sweeps(nsweeps, len(y))

    bins = generate_bins(y, bins)

    if sigma is None:
        if np.isscalar(sbins):
            sbins = np.linspace(1e-2, 4, sbins)
        
        # Estimate the density using a range of kernel bandwidths
        results = [estimate_mixture(y, f0, sigma=s, sweeporder=sweeporder, bins=bins, **kwargs) for s in sbins]

        # Calculate the 'predictive recursion marginal likelihood' (PRML) of y for each bandwidth
        marginals = [r['logm'] for r in results]

        # Weight each density proportional to the PRML of y
        Zs = [r['z'] for r in results]
        z_logits = np.exp(marginals - np.max(marginals))
        z_probs = z_logits / z_logits.sum()
        z_hat = (z_probs[:,None] * Zs).sum(axis=0)

        # Create the new weighted distribution
        result =  results[np.argmax(marginals)]
        result['dist'] = GridDistribution1D(bins, z_hat)
        result['z'] = z_hat
        result['logm_grid'] = marginals
        result['w'] = (z_probs[:,None] * [r['w'] for r in results]).sum(axis=0)

        return result

    if weights is None:
        # Assign equal weight to every data point
        weights = np.ones_like(y)

    if kernel is None or kernel == 'normal':
        kernel = lambda x, b, s: norm.pdf(x[:,None], b[None], scale=s)

    # Initialize everything to mostly the null distribution
    w = np.ones(len(bins)+1)
    w[0] = 0.99
    w[1:] = 0.01 / (bins.max() - bins.min())

    # Calculate the likelihood of each z under the null
    null_likelihoods = f0.pdf(y)

    # Calculate the likelihood of each z coming from a N(mu, 1) centered at this bin
    alt_likelihoods = kernel(y, bins, sigma)

    # Sweep through the data and reweight the density using each point iteratively
    log_marginal = 0
    for i, k in enumerate(sweeporder):
        step_weight = (3. + i)**decay # Each iteration contributes slightly less
        f0 = w[0]*null_likelihoods[k] # null likelihood of z_k, weighted by prior w0
        f1 = w[1:]*alt_likelihoods[k] # alt likelihood of z_k, weighted by prior w1
        m = np.trapz(f1, bins) + f0
        if i < len(y):
            log_marginal += np.log(max(1e-10, m))
        w[0] = (1 - step_weight * weights[k]) * w[0] + step_weight * weights[k] * f0 / m
        w[1:] = (1 - step_weight * weights[k]) * w[1:] + step_weight * weights[k] * f1 / m

    # Get the marginal likelihood of each point to create a grid approximation
    z = (w[None,1:]*kernel(bins, bins, sigma)).sum(axis=1)
    z /= 1-w[0]
    z /= np.trapz(z, bins)

    return {'bins': bins, 'dist': GridDistribution1D(bins, z), 'w': w,
             'z': z, 'logm': log_marginal, 'sweeporder': sweeporder,
             'f1': GridDistribution1D(bins, w[1:])}

if __name__ == '__main__':
    # test_1d_estimates()
    nbins = 50
    z = np.concatenate([np.random.normal(0.3, 1.2, size=250), np.random.normal(-2, 2, size=50)])
    f0 = one_sided_empirical_null(z)
    mix = estimate_mixture(z, f0)
    f1 = mix['dist']

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        counts, bins, _ = plt.hist(z, bins=nbins, color='gray', alpha=0.5)

        plt.plot(f0.bins, f0.w*counts.max() / f0.w.max(), lw=3, color='blue', label='Empirical null')
        plt.plot(f1.bins, f1.w*counts.max() / f1.w.max(), lw=3, color='orange', label='Alternative')

        # true_probs = norm.pdf(null_grid, 0.3, scale=1.2)
        # true_probs /= true_probs.sum()
        # plt.plot(null_grid, true_probs*(z >= mid_bin).sum()*2, lw=3, label="Truth")
        plt.xlabel('Test statistics', fontsize=18, weight='bold', fontname='Times New Roman')
        plt.ylabel('Counts', fontsize=18, weight='bold', fontname='Times New Roman')
        plt.legend(loc='upper right')
        plt.savefig('plots/empirical-bayes-knockoff-null.pdf', bbox_inches='tight')
    plt.close()


