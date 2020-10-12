import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
from normix import GridDistribution1D
from cancer_dataset import DrugData, load_drug_data, load_drug_ids

def load_results(direc, idx):
    try:
        h_predictions = np.load(f'data/{direc}/h_predictions_{idx}.npy')
        bh_preds = np.load(f'data/{direc}/bh_predictions_{idx}.npy')
        h_crt = np.load(f'data/{direc}/feature_predictions_{idx}.npy')
        h_crt = np.load(f'data/{direc}/feature_predictions_{idx}.npy')
        knockoff_preds = np.load(f'data/{direc}/knockoff_predictions_{idx}.npy')
        null_dist_bins, null_dist_w = np.load(f'data/{direc}/knockoff_empirical_null_{idx}.npy')
        null_dist = GridDistribution1D(null_dist_bins, null_dist_w)
    except:
        print('No results found')
        return None

    try:
        h1_true = np.load(f'data/{direc}/h_true_{idx}.npy')
        signal_indices = np.load(f'data/{direc}/signal_indices_{idx}.npy')
        h2_true = np.zeros(h_crt.shape, dtype=bool)
        h2_true[signal_indices] = True
    except:
        h1_true, h2_true = None, None

    return h_predictions, bh_preds, h_crt, knockoff_preds, null_dist, h1_true, h2_true

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot the results for the case study applying DEBT to a cancer drug screening dataset.')
    parser.add_argument('direc', help='The directory with the results.')

    
    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Specify the name of the drug -- either lapatinib or nutlin
    import glob
    import re
    indices = [int(re.search('\\d+', f).group(0)) for f in glob.glob(f'data/{args.direc}/h_predictions_*.npy')]
    print(indices)
    results = {'h1-debt': [], 'h1-bh': [], 'h2-debt': [], 'h2-ko': [],
               'h1-debt-fdp': [], 'h1-bh-fdp': [], 'h2-debt-fdp': [], 'h2-ko-fdp': [],
               'fdp': {'Method': [], 'FDP': []}}
    for loop_idx, trial_idx in enumerate(indices):
        print('Index {}/{}'.format(loop_idx+1, len(indices)))
        drug_results = load_results(args.direc, trial_idx)
        if drug_results is None:
            continue
        h1_debt, h1_bh, h2_debt, h2_ko, crt_null_dist, h1_true, h2_true = drug_results
        results['h1-debt'].append(h1_debt.sum())
        results['h1-bh'].append(h1_bh.sum())
        results['h2-debt'].append(h2_debt.sum())
        results['h2-ko'].append(h2_ko.sum())
        if h1_true is not None:
            results['h1-debt-fdp'].append(((~h1_true) & h1_debt).sum() / max(1, h1_debt.sum()))
            results['h1-bh-fdp'].append(((~h1_true) & h1_bh).sum() / max(1, h1_bh.sum()))
            results['h2-debt-fdp'].append(((~h2_true) & h2_debt).sum() / max(1, h2_debt.sum()))
            results['h2-ko-fdp'].append(((~h2_true) & h2_ko).sum() / max(1, h2_ko.sum()))
            results['fdp']['Method'].extend(['DEBT\n(Stage 1)', 'BH', 'DEBT\n(Stage 2)', 'Knockoffs'])

    print('Total results: {}'.format(len(results['h1-debt'])))
    # Plot the results in comparison to BH
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 18
        matplotlib.rcParams['axes.labelsize'] = 20
        matplotlib.rcParams['legend.fontsize'] = 18


        plt.scatter(results['h1-bh'], results['h1-debt'], color='0.1')
        plt.plot([np.min([results['h1-bh'], results['h1-debt']]), np.max([results['h1-bh'], results['h1-debt']])],
                 [np.min([results['h1-bh'], results['h1-debt']]), np.max([results['h1-bh'], results['h1-debt']])],
                 color='black', lw=2)
        plt.xlabel('Benjamini-Hochberg', fontsize=18, weight='bold')
        plt.ylabel('DEBT', fontsize=18, weight='bold')
        plt.savefig(f'plots/{args.direc}-stage1-comparison.pdf', bbox_inches='tight')
        plt.close()

        x, y = np.array(results['h2-ko']), np.array(results['h2-debt'])
        mask = (x != 0) | (y != 0)
        x, y = x[mask], y[mask]

        binrange = np.min([x, y]), np.max([x, y])
        hist, xbins,ybins = np.histogram2d(y,x, bins=np.arange(binrange[0], binrange[1]+1))
        X,Y = np.meshgrid(xbins[:-1], ybins[:-1])
        X = X[hist != 0]; Y = Y[hist != 0]
        Z = hist[hist != 0]

        fig, ax = plt.subplots()
        ax.imshow(hist, cmap='gray_r', vmin=0, vmax=Z.max()*1.3)
        for i in range(len(Z)):
            ax.annotate(str(int(Z[i])), xy=(X[i],Y[i]), xytext=(0,0), color='black',
                        ha='center', va='center', textcoords='offset points')
        plt.plot(binrange, binrange, color='black', lw=1)
        plt.xlabel('Knockoffs', fontsize=18, weight='bold')
        plt.ylabel('DEBT', fontsize=18, weight='bold')
        plt.gca().invert_yaxis()
        plt.savefig(f'plots/{args.direc}-stage2-comparison.pdf', bbox_inches='tight')
        plt.close()

        # Look at the cumulative curves
        grid = np.arange(binrange[1]+1)
        xgrid = (x[:,None] >= grid[None]).sum(axis=0)
        ygrid = (y[:,None] >= grid[None]).sum(axis=0)
        plt.plot(grid, xgrid, label='Knockoffs', color='gray')
        plt.plot(grid, ygrid, label='DEBT', color='black')
        plt.xlabel('Minimum discoveries', fontsize=18, weight='bold')
        plt.ylabel('# experiments', fontsize=18, weight='bold')
        plt.legend(loc='upper right')
        plt.savefig(f'plots/{args.direc}-stage2-comparison-cumulative.pdf', bbox_inches='tight')
        plt.close()

        # Plot FDR results if we have ground truth info
        if h1_true is not None:
            df = {'Method': [], 'FDP': []}

            df['Method'].extend(['DEBT\n(Stage 1)']*len(results['h1-debt-fdp']))
            df['Method'].extend(['BH']*len(results['h1-bh-fdp']))
            df['Method'].extend(['DEBT\n(Stage 2)']*len(results['h2-debt-fdp']))
            df['Method'].extend(['Knockoffs']*len(results['h2-ko-fdp']))

            df['FDP'].extend(results['h1-debt-fdp'])
            df['FDP'].extend(results['h1-bh-fdp'])
            df['FDP'].extend(results['h2-debt-fdp'])
            df['FDP'].extend(results['h2-ko-fdp'])

            ax = sns.boxplot(x='Method', y='FDP', data=pd.DataFrame(df), palette=['w']*4, color='black', flierprops=dict(markerfacecolor='black', markersize=12))
            # Iterate over boxes
            for i,box in enumerate(ax.artists):
                box.set_edgecolor('black')
                box.set_facecolor('white')

                # iterate over whiskers and median lines
                for j in range(6*i,6*(i+1)):
                     ax.lines[j].set_color('black')

            plt.axhline(0.2, ls='--', color='black')
            plt.xlabel('')
            plt.ylabel('FDP', fontsize=18, weight='bold')
            plt.savefig(f'plots/{args.direc}-fdp.pdf', bbox_inches='tight')
            plt.close()




