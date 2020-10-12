'''Case study of applying DEBT to a cancer drug screening.

The data are taken from the Genomics of Drug Sensitivity in Cancer (GDSC). We
preprocessed the max dosage readings and converted them into into z-scores.

DEBT performs two stages of selection.

Stage 1: Fit a (black box) neural network model using the mutation data for each
cell line. The NN is then used to select the significant outcomes at a given
false discovery rate (FDR). We use the default 10% FDR.

Stage 2: Use an empirical Bayes extension of model-X knockoffs to selection
important features at a given FDR. We again use 10% FDR.

Note that the code is setup to be run easily on a cluster, in case one has
hundreds of such studies to analyze. The model is checkpointed frequently
in order to allow for preemption without losing the progress. The whole process
should finish in about 5-10 minutes on a modern laptop.
'''
from __future__ import print_function
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
from utils import p_value_2sided, bh_predictions

if __name__ == '__main__':
    np.random.seed(42)
    fdr = 0.1

    # Specify the name of the drug -- either lapatinib or nutlin
    drug = sys.argv[1]

    print('Loading data for drug={}'.format(drug))
    sys.stdout.flush()
    X = np.load('data/cancer/{}_x.npy'.format(drug))
    z = np.load('data/cancer/{}_z.npy'.format(drug)).clip(-10,10) # Clip at +/- 10 since that's effectively zero prob null
    gene_names = np.load('data/cancer/{}_genes.npy'.format(drug))
    cancer_types = np.load('data/cancer/{}_types.npy'.format(drug))

    print('nsamples: {} nfeatures: {}'.format(X.shape[0], X.shape[1]))
    sys.stdout.flush()

    #### Two-groups empirical bayes model ####
    print('Creating blackbox 2-groups model')
    fdr_model = BlackBoxTwoGroupsModel(X, z, fdr=fdr)

    print('Training')
    sys.stdout.flush()
    results = fdr_model.train(save_dir='data/cancer/{}_twogroups'.format(drug),
                              verbose=False, batch_size=60 if X.shape[0] > 1000 else 10,
                              num_folds=10, num_epochs=100)

    # Save the Stage 1 significant experimental outcome results
    h_predictions = results['predictions']
    h_posteriors = results['posteriors']
    bh_preds = bh_predictions(p_value_2sided(z), fdr)
    
    #### Posterior EB knockoffs model ####
    print('Model not found. Creating a new one.')
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
    knockoff_stats = true_tstat - crt_results['tstats']
    knockoff_preds[knockoff_filter(knockoff_stats, fdr, offset=0)] = True

    # Save everything to file
    np.save('data/cancer/h_posteriors_{}.npy'.format(drug), h_posteriors)
    np.save('data/cancer/h_predictions_{}.npy'.format(drug), h_predictions)
    np.save('data/cancer/bh_predictions_{}.npy'.format(drug), bh_preds)
    np.save('data/cancer/feature_predictions_{}.npy'.format(drug), h_crt)
    np.save('data/cancer/feature_tstats_{}.npy'.format(drug), t_crt)
    np.save('data/cancer/knockoff_predictions_{}.npy'.format(drug), knockoff_preds)
    np.save('data/cancer/knockoff_stats_{}.npy'.format(drug), knockoff_stats)

    print('Significant genes:')
    for g,pred in zip(gene_names,h_crt.astype(bool)):
        if pred:
            print(g)

    print('Genes unique to DEBT:')
    for g,pred in zip(gene_names,(h_crt.astype(bool) & ~knockoff_preds.astype(bool))):
        if pred:
            print(g)
    
    print('DEBT discoveries: {}'.format(h_predictions.sum()))
    print('BH discoveries:     {}'.format(bh_preds.sum()))
    print('')

    print('DEBT features:    {}'.format(crt_results['discoveries'].sum()))
    
    print('knockoffs features: {}'.format(knockoff_preds.sum()))

    # Plot the results in comparison to BH
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        bins = np.linspace(-10,2,30)
        plt.hist(z[~h_predictions.astype(bool)], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(z[h_predictions.astype(bool)], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        plt.xlabel('z', fontsize=36)
        plt.ylabel('Count', fontsize=36)
        plt.savefig('plots/debt-{}-stage1.pdf'.format(drug), bbox_inches='tight')
        plt.close()

        plt.hist(z[~bh_preds.astype(bool)], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(z[bh_preds.astype(bool)], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        if drug == 'lapatinib':
            legend_props = {'weight': 'bold', 'size': 28}
            plt.legend(loc='upper left', prop=legend_props)
        plt.xlabel('z', fontsize=36)
        plt.ylabel('Count', fontsize=36)
        plt.savefig('plots/bh-{}-stage1.pdf'.format(drug), bbox_inches='tight')
        plt.close()

        from normix import generate_bins
        bins = generate_bins(t_crt, 50)
        plt.hist(t_crt[~h_crt.astype(bool)], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(t_crt[h_crt.astype(bool)], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        plt.xlabel('t', fontsize=36)
        plt.ylabel('Count', fontsize=36)
        plt.savefig('plots/debt-{}-stage2.pdf'.format(drug), bbox_inches='tight')
        plt.close()

        plt.hist(t_crt[~knockoff_preds], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(t_crt[knockoff_preds], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        plt.xlabel('t', fontsize=36)
        plt.ylabel('Count', fontsize=36)
        plt.savefig('plots/knockoffs-{}-stage2.pdf'.format(drug), bbox_inches='tight')
        plt.close()
        




