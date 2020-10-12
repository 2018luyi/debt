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
from synthetic import generate_synthetic_z

def run_analysis(drug=1, fdr=0.2, save_drug_data=False, synthetic=False, **kwargs):
    np.random.seed(42)
    torch.manual_seed(42)
        
    print('Loading data for drug={}'.format(drug))
    sys.stdout.flush()
    data = load_drug_data(drug, save=save_drug_data)
    X, z, feature_names, cancer_types = data.X, data.z.clip(-10,10), data.feature_names, data.cancer_types
    
    print('drug name: {} nsamples: {} nfeatures: {}'.format(data.drug_name, X.shape[0], X.shape[1]))
    sys.stdout.flush()
    if synthetic:
        print('Getting synthetic response')
        z, h_true, signal_indices, coefficients = generate_synthetic_z(X, **kwargs)
        if h_true.sum() == 0:
            print('WARNING: no signals!')
            raise Exception()
        print(f'{h_true.sum()} true positives and {len(signal_indices)} nonnull features.')
    
    direc = 'synthetic' if synthetic else 'cancer'
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
    np.save(f'data/{direc}/h_predictions_{drug}.npy', h_predictions)
    np.save(f'data/{direc}/bh_predictions_{drug}.npy', bh_preds)
    np.save(f'data/{direc}/feature_predictions_{drug}.npy', h_crt)
    np.save(f'data/{direc}/h_priors_{drug}.npy', fdr_model.priors)
    np.save(f'data/{direc}/h_posteriors_{drug}.npy', fdr_model.posteriors)
    np.save(f'data/{direc}/z_empirical_null_{drug}.npy', fdr_model.null_dist)
    np.save(f'data/{direc}/z_alternative_{drug}.npy', [fdr_model.alt_dist.x, fdr_model.alt_dist.y])
    np.save(f'data/{direc}/knockoff_predictions_{drug}.npy', knockoff_preds)
    np.save(f'data/{direc}/knockoff_empirical_null_{drug}.npy', [crt.null_dist.bins, crt.null_dist.w])
    np.save(f'data/{direc}/knockoff_alternative_{drug}.npy', [crt.alt_dist.bins, crt.alt_dist.w])
    np.save(f'data/{direc}/knockoff_prior_{drug}.npy', [crt.pi0])
    np.save(f'data/{direc}/knockoff_posteriors_{drug}.npy', crt.posteriors)

    if synthetic:
        # Save the ground truth
        np.save(f'data/{direc}/h_true_{drug}.npy', h_true)
        np.save(f'data/{direc}/signal_indices_{drug}.npy', signal_indices)
        np.save(f'data/{direc}/signal_coefs_{drug}.npy', coefficients)

    print('Significant genes:')
    for g,pred in zip(feature_names,h_crt.astype(bool)):
        if pred:
            print(g)

    print('Genes unique to DEBT:')
    for g,pred in zip(feature_names,(h_crt.astype(bool) & ~knockoff_preds.astype(bool))):
        if pred:
            print(g)
    
    if synthetic:
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
    else:
        print('')
        print('DEBT discoveries: {}'.format(h_predictions.astype(int).sum()))
        print('BH discoveries:     {}'.format(bh_preds.astype(int).sum()))
        print('DEBT features:    {}'.format(h_crt.astype(int).sum()))
        print('knockoffs features: {}'.format(knockoff_preds.astype(int).sum()))

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
        vals, histbins, patches = plt.hist(t_crt[~h_crt.astype(bool)], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(t_crt[h_crt.astype(bool)], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        plt.plot(crt.null_dist.bins, crt.null_dist.w*max(vals)*2, lw=2, label='Empirical null', color='gray')
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
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Case study applying DEBT to a cancer drug screening dataset.')

    # Experiment settings
    parser.add_argument('--save', action='store_true', default=False, help='Save the results of each drug trial to file.')
    parser.add_argument('--synthetic', action='store_true', default=False, help='Run a semi-synthetic experiment where the ground truth is known.')
    parser.add_argument('--drug', type=int, help='ID of a single drug to run. If not specified, runs all drugs.')
    parser.add_argument('--start', type=int, default=0, help='Index to start at.')
    parser.add_argument('--end', type=int, default=-1, help='Index to end at (-1 = last).')
    parser.add_argument('--fdr', type=float, default=0.2, help='The nominal false discovery rate to target in both stages.')
    parser.add_argument('--nsignals', type=int, default=40, help='The number of nonnull features.')
    parser.add_argument('--signal_strength', type=float, default=3, help='The average signal strength.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Specify the name of the drug -- either lapatinib or nutlin
    if args.drug is not None:
        run_analysis(save_drug_data=args.save, **dargs)
    else:
        drug_ids = load_drug_ids()
        for idx, drug in enumerate(drug_ids):
            if idx < args.start:
                continue
            if idx > args.end and args.end != -1:
                break
            try:
                print('Drug {}/{}'.format(idx+1, len(drug_ids)))
                dargs['drug'] = drug
                run_analysis(save_drug_data=args.save, **dargs)
                print('\n\n')
            except:
                print('Crashed. Skipping...')

    




