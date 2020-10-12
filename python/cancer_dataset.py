import numpy as np
import pandas as pd
from utils import pav

class DrugData:
    def __init__(self, drug_id, drug_name, z, X, feature_names, cell_names, cancer_types):
        self.drug_id = drug_id
        self.drug_name = drug_name
        self.z = z
        self.X = X
        self.feature_names = feature_names
        self.cell_names = cell_names
        self.cancer_types = cancer_types

def load_drug_ids():
    df = pd.read_csv('data/cancer/drug_details.csv', header=0)
    return df['Drug ID'].values

def load_drug_data(drug_id, save=False):
    df = pd.read_csv('data/cancer/drug_details.csv', header=0)
    drug_name = df.iloc[np.where(df['Drug ID'] == drug_id)[0]]['Drug Name']
    if not isinstance(drug_name, str):
        drug_name = drug_name.values[0]
    
    print('Loading data')
    experiments = pd.read_csv('data/cancer/responses.csv', header=0)
    features = pd.read_csv('data/cancer/features.csv',delimiter=',',header=0,index_col=0).T
    types = {row['sample-name']: row['gdsc-tissue-desc-1'] for i,row in pd.read_csv('data/cancer/types.csv').iterrows()}

    cell_lines = list(features.columns)
    feature_names = list(features.index)

    gene_features = lambda line: features.iloc[:,cell_lines.index(line)].values.astype(int) if line in cell_lines else None
    X = []
    z = []
    Types = []
    Cell_names = []
    print('Populating the feature matrices')
    for i,row in experiments[experiments['DRUG_ID'] == drug_id].iterrows():
        x = gene_features(row['CELL_LINE_NAME'])
        if x is None:
            continue

        X.append(x)
        Cell_names.append(row['CELL_LINE_NAME'])

        if row['CELL_LINE_NAME'] in types:
            Types.append('{} ({})'.format(row['CELL_LINE_NAME'], types[row['CELL_LINE_NAME']]))
        else:
            Types.append('{} ({})'.format(row['CELL_LINE_NAME'], 'unknown'))

        # Choose the right z based on whether it was 2x or 4x dilution
        if np.isnan(row['z_dose8']):
            z.append(pav(row[['z_dose0','z_dose1','z_dose2','z_dose3','z_dose4']].values.astype(float)))
        else:
            z.append(pav(row[['z_dose0','z_dose2','z_dose4','z_dose6','z_dose8']].values.astype(float)))
        
    # Create the arrays
    X = np.array(X)
    z = np.array(z)
    Types = np.array(Types)
    Cell_names = np.array(Cell_names)

    # Choose the highest dose z-scores
    print(z.shape)
    max_var = np.argmax(z.clip(-5,5).astype(int).std(axis=0))
    z = z[:,max_var] # maximum variance
    print('Dose chosen: {}'.format(max_var))

    variable_genes = (X.std(axis=0) != 0)
    X = X[:,variable_genes]
    print('Drug: {} X: {} z: {}'.format(drug_name, X.shape, z.shape))
    if save:
        np.save('data/cancer/{}_x'.format(drug_id), X)
        np.save('data/cancer/{}_z'.format(drug_id), z)
        np.save('data/cancer/{}_genes'.format(drug_id), np.array([g for g,v in zip(feature_names,variable_genes) if v]))
        np.save('data/cancer/{}_types'.format(drug_id), Types)
        np.save('data/cancer/{}_cell_lines'.format(drug_id), Cell_names)
    return DrugData(drug_id, drug_name, z, X, feature_names, Cell_names, Types)

if __name__ == '__main__':
    '''Plots the results so far.'''
    import matplotlib
    import matplotlib.pylab as plt
    import seaborn as sns
    drug_ids = load_drug_ids()
    stage1_debt = []
    stage1_bh = []
    stage2_debt = []
    stage2_knockoffs = []
    for drug in drug_ids:
        try:
            h_predictions = np.load('data/cancer/h_predictions_{}.npy'.format(drug))
            bh_predictions = np.load('data/cancer/bh_predictions_{}.npy'.format(drug))
            h_crt = np.load('data/cancer/feature_predictions_{}.npy'.format(drug))
            knockoff_preds = np.load('data/cancer/knockoff_predictions_{}.npy'.format(drug))
        except:
            continue

        stage1_debt.append(h_predictions.sum())
        stage1_bh.append(bh_predictions.sum())
        stage2_debt.append(h_crt.sum())
        stage2_knockoffs.append(knockoff_preds.sum())

    stage1_debt = np.array(stage1_debt)
    stage1_bh = np.array(stage1_bh)
    stage2_debt = np.array(stage2_debt)
    stage2_knockoffs = np.array(stage2_knockoffs)

    # Plot the results in comparison to BH
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        plt.hist(stage1_debt - stage1_bh, color='gray', alpha=0.5, bins=20)
        plt.xlabel('DEBT gain over BH', fontsize=30)
        plt.ylabel('Count', fontsize=30)
        plt.savefig('plots/stage1-gain.pdf', bbox_inches='tight')
        plt.close()

        plt.hist(stage2_debt - stage2_knockoffs, color='gray', alpha=0.5, bins=20)
        plt.xlabel('DEBT gain over knockoffs', fontsize=30)
        plt.ylabel('Count', fontsize=30)
        plt.savefig('plots/stage2-gain.pdf', bbox_inches='tight')
        plt.close()




















