'''Synthetic benchmark comparing DEBT to BH and knockoffs.
'''
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import sys
import os
import torch
import torch.nn as nn
from two_groups_beta import BlackBoxTwoGroupsModel
from posterior_crt import PosteriorConditionalRandomizationTester
from utils import p_value_2sided, bh_predictions, ilogit
from cancer_dataset import DrugData, load_drug_data, load_drug_ids

def generate_synthetic_z(X, nsignals=10, signal_strength=1, alt_strength=-2, max_nonnull=0.4, **kwargs):
    # Generate responses under the constraint that not too many are nonnull
    h = np.ones(X.shape[0])
    while h.mean() > max_nonnull:
        # Sample the true nonnull features
        signal_indices = np.random.choice(X.shape[1], replace=False, size=nsignals)

        # Random coefficients with an average signal strength
        beta = np.random.normal(0, signal_strength/np.sqrt(nsignals+nsignals//2), size=nsignals + nsignals//2)
        np.set_printoptions(precision=2, suppress=True)

        # Quadratic interaction function
        logits = (X[:,signal_indices].dot(beta[:nsignals]) +
                    (X[:,signal_indices[:nsignals//2]] * X[:,signal_indices[nsignals//2:nsignals//2*2]]).dot(beta[nsignals:]) - 1)

        # Get whether this was a signal or not
        r = np.random.random(size=logits.shape[0])
        h = ilogit(logits) >= r
        
        # If it was, get the z-score
        z = np.random.normal(alt_strength * h, 1).clip(-10,10)

    return z, h, signal_indices, beta

def generate_synthetic_X(nsamples=1000, nfeatures=100, r=0.25, **kwargs):
    Sigma = np.array([np.exp(-np.abs(np.arange(nfeatures) - i)) * r for i in range(nfeatures)])
    logits = np.random.multivariate_normal(np.zeros(nfeatures), Sigma, size=nsamples)
    X = (np.random.random(size=logits.shape) <= ilogit(logits)).astype(int)
    return X

def run_analysis(idx, fdr=0.2, **kwargs):
    np.random.seed(8+idx)
    torch.manual_seed(8+idx)
        
    print('Generating synthetic covariates')
    X = generate_synthetic_X(**kwargs)
        
    print('Getting synthetic response')
    z, h_true, signal_indices, coefficients = generate_synthetic_z(X, **kwargs)
    if h_true.sum() == 0:
        print('WARNING: no signals!')
        raise Exception()
    
    print(f'{h_true.sum()} true positives, {(~h_true).sum()} nulls, and {len(signal_indices)} nonnull features.')

    direc = 'pure-synthetic'
    if not os.path.exists(f'data/{direc}'):
        os.makedirs(f'data/{direc}')

    #### Two-groups empirical bayes model ####
    print('Stage 1: Creating blackbox 2-groups model')
    fdr_model = BlackBoxTwoGroupsModel(X, z, fdr=fdr, estimate_null=True)

    print('Training')
    sys.stdout.flush()
    results = fdr_model.train(save_dir=f'data/{direc}/twogroups',
                              verbose=False, batch_size=None,
                              num_folds=5, num_epochs=100)

    # Save the Stage 1 significant experimental outcome results
    h_predictions = results['predictions']
    bh_preds = bh_predictions(p_value_2sided(z, mu0=fdr_model.null_dist[0], sigma0=fdr_model.null_dist[1]), fdr)
    
    #### Posterior EB knockoffs model ####
    print('Stage 2: Empirical Bayes knockoffs')
    crt = PosteriorConditionalRandomizationTester(fdr_model, fdr=fdr)
    
    # Run a CRT for each feature
    crt.run(verbose=False)

    print('Getting the predictions')
    crt_results = crt.predictions()
    t_crt = crt_results['tstats']
    h_crt = crt_results['discoveries']

    # Try the model-X knockoffs approach using the same knockoff samples
    from knockoffs import knockoff_filter
    knockoff_preds = np.zeros(len(crt.tstats), dtype=bool)
    true_tstat = (fdr_model.posteriors*np.log(fdr_model.posteriors) + (1-fdr_model.posteriors)*np.log(1-fdr_model.posteriors)).mean()
    true_tstat -= crt.tstats_mean
    true_tstat /= crt.tstats_std
    plt.hist(true_tstat - crt.tstats, bins=30)
    plt.savefig('plots/knockoffs-temp.pdf', bbox_inches='tight')
    plt.close()
    knockoff_preds[knockoff_filter(true_tstat - crt_results['tstats'], fdr, offset=1.0)] = True

    # Save everything to file
    np.save(f'data/{direc}/h_predictions_{idx}.npy', h_predictions)
    np.save(f'data/{direc}/bh_predictions_{idx}.npy', bh_preds)
    np.save(f'data/{direc}/feature_predictions_{idx}.npy', h_crt)
    np.save(f'data/{direc}/h_priors_{idx}.npy', fdr_model.priors)
    np.save(f'data/{direc}/h_posteriors_{idx}.npy', fdr_model.posteriors)
    np.save(f'data/{direc}/z_empirical_null_{idx}.npy', fdr_model.null_dist)
    np.save(f'data/{direc}/z_alternative_{idx}.npy', [fdr_model.alt_dist.x, fdr_model.alt_dist.y])
    np.save(f'data/{direc}/knockoff_predictions_{idx}.npy', knockoff_preds)
    np.save(f'data/{direc}/knockoff_empirical_null_{idx}.npy', [crt.null_dist.bins, crt.null_dist.w])
    np.save(f'data/{direc}/knockoff_alternative_{idx}.npy', [crt.alt_dist.bins, crt.alt_dist.w])
    np.save(f'data/{direc}/knockoff_prior_{idx}.npy', [crt.pi0])
    np.save(f'data/{direc}/knockoff_posteriors_{idx}.npy', crt.posteriors)

    # Save the ground truth
    np.save(f'data/{direc}/h_true_{idx}.npy', h_true)
    np.save(f'data/{direc}/signal_indices_{idx}.npy', signal_indices)
    np.save(f'data/{direc}/signal_coefs_{idx}.npy', coefficients)
    
    # Report results
    tpp1_debt = (h_true & h_predictions).sum() / max(1, h_true.sum())
    tpp1_bh = (h_true & bh_preds).sum() / max(1, h_true.sum())
    tpp2_debt = len([s for s in signal_indices if h_crt[s]]) / len(signal_indices)
    tpp2_knockoffs = len([s for s in signal_indices if knockoff_preds[s]]) / len(signal_indices)

    fdp1_debt = ((~h_true) & h_predictions).sum() / max(1, h_predictions.sum())
    fdp1_bh = ((~h_true) & bh_preds).sum() / max(1, bh_preds.sum())
    fdp2_debt = len([s for s, h in enumerate(h_crt) if h and s not in signal_indices]) / max(1,h_crt.sum())
    fdp2_knockoffs = len([s for s, h in enumerate(knockoff_preds) if h and s not in signal_indices]) / max(1,knockoff_preds.sum())
    print('')
    print(f'DEBT discoveries:   {h_predictions.sum()} TPP: {tpp1_debt*100:.2f}% FDP: {fdp1_debt*100:.2f}%')
    print(f'BH discoveries:     {bh_preds.sum()} TPP: {tpp1_bh*100:.2f}% FDP: {fdp1_bh*100:.2f}%')
    print(f'DEBT features:      {h_crt.sum()} TPP: {tpp2_debt*100:.2f}% FDP: {fdp2_debt*100:.2f}%')
    print(f'Knockoffs features: {knockoff_preds.sum()} TPP: {tpp2_knockoffs*100:.2f}% FDP: {fdp2_knockoffs*100:.2f}%')
    
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Synthetic benchmarks for DEBT versus BH and knockoffs.')

    # Experiment settings
    parser.add_argument('--fdr', type=float, default=0.2, help='The nominal false discovery rate to target in both stages.')
    parser.add_argument('--ntrials', type=int, default=100, help='Number of independent trials to run.')
    parser.add_argument('--nsamples', type=int, default=500, help='The number of iid samples of (x,z).')
    parser.add_argument('--nfeatures', type=int, default=100, help='The number of total features.')
    parser.add_argument('--nsignals', type=int, default=20, help='The number of nonnull features.')
    parser.add_argument('--signal_strength', type=float, default=3, help='The average signal strength.')
    parser.add_argument('--r', type=float, default=1, help='The coefficient of variation.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    for idx in range(args.ntrials):
        print('Trial {}/{}'.format(idx+1, args.ntrials))
        dargs['idx'] = idx
        run_analysis(**dargs)
        print('\n\n')

    




