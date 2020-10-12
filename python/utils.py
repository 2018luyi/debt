import numpy as np
from scipy import stats as st

def p2z(p):
    return st.norm.ppf(p)

def fdr(truth, pred, axis=None):
    return ((pred==1) & (truth==0)).sum(axis=axis) / pred.sum(axis=axis).astype(float).clip(1,np.inf)

def tpr(truth, pred, axis=None):
    return ((pred==1) & (truth==1)).sum(axis=axis) / truth.sum(axis=axis).astype(float).clip(1,np.inf)

def true_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==1)).sum(axis=axis)

def false_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==0)).sum(axis=axis)

def ilogit(x):
    return 1. / (1. + np.exp(-x))


def pretty_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, ignore)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, ignore, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, ignore) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, ignore=None):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if ((hasattr(ignore, "__len__") and a in ignore) or a == ignore) else style.format(a) for a in p]))


def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b*batch_size:min((b+1)*batch_size, len(order))]
        yield idx


def p_value_2sided(z, mu0=0., sigma0=1.):
    return 2*(1.0 - st.norm.cdf(np.abs((z - mu0) / sigma0)))

def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries)

def bh_predictions(p_values, fdr_threshold):
    pred = np.zeros(len(p_values), dtype=int)
    disc = bh(p_values, fdr_threshold)
    if len(disc) > 0:
        pred[disc] = 1
    return pred

def sample_gmm(pi, mu, sigma):
    p = np.random.random()
    for pi_i, mu_i, sigma_i in zip(np.cumsum(pi), mu, sigma):
        if p < pi_i:
            return np.random.normal(mu_i, sigma_i)
    return np.random.normal(mu[-1], sigma[-1])

def gmm_pdf(x, pi, mu, sigma):
    return np.array([pi_i * st.norm.pdf(x, mu_i, sigma_i) for pi_i, mu_i, sigma_i in zip(pi, mu, sigma)]).sum(axis=0)

def create_folds(X, k):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) / k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[int(start):int(end)])
    return folds

def calc_fdr(probs, fdr_level):
    '''Calculates the detected signals at a specific false discovery rate given the posterior probabilities of each point.'''
    pshape = probs.shape
    if len(probs.shape) > 1:
        probs = probs.flatten()
    post_orders = np.argsort(probs)[::-1]
    avg_fdr = 0.0
    end_fdr = 0
    
    for idx in post_orders:
        test_fdr = (avg_fdr * end_fdr + (1.0 - probs[idx])) / (end_fdr + 1.0)
        if test_fdr > fdr_level:
            break
        avg_fdr = test_fdr
        end_fdr += 1

    is_finding = np.zeros(probs.shape, dtype=int)
    is_finding[post_orders[0:end_fdr]] = 1
    if len(pshape) > 1:
        is_finding = is_finding.reshape(pshape)
    return is_finding

