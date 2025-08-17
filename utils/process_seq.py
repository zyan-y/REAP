import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold

# read seq and name to generate .fa files for each
def excel2fa(excel_file, fa_file):
    df = pd.read_excel(excel_file, header=0, usecols=[0,1])
    with open(fa_file, 'w') as f:
        for _, row in df.iterrows():
            seq = row['seq']
            name = row['name']
            f.write(f'>{name}\n')
            f.write(f'{seq}\n')

# read mut seqs to generate mut names
def seqs_to_short(file, first_wt=True, wt_file=''):
    data = pd.read_excel(file, usecols=[0,1]).values
    seq_wt = data[0][1] if first_wt else str(SeqIO.read(wt_file, 'fasta').seq)
    muts_seq = []
    for name, seq_mut in data[1:]:
        muts = []
        for i in range(len(seq_mut)):
            if seq_mut[i] != seq_wt[i]:
                mutstr = f'{seq_wt[i]}{i+1}{seq_mut[i]}'
                muts.append(mutstr)
        muts = '-'.join(muts)
        muts_seq.append([name, muts])
    df = pd.DataFrame(muts_seq)
    df.to_excel('muts_seq_short.xlsx', index=False, header=False)

# read mut names to generate mut seqs
def short_to_seqs(data, seq_wt, save_fa=True, save_fa_each=False, save_excel=False, folder='', delimiter='-'):
    muts_seqs = []
    for name in data:
        shorts = name.split(delimiter)
        mut_seq = seq_wt
        for short in shorts:
            wt, pos, mut = short[0], int(short[1:-1])-1, short[-1]
            if seq_wt[pos] != wt:
                print(f'Wrong! {name}')
                break
            mut_seq = mut_seq[0:pos] + mut + mut_seq[pos+1:]
        muts_seqs.append([name, mut_seq])
    if save_excel:
        df = pd.DataFrame(muts_seqs, columns=['name', 'seq'])
        df.to_excel('muts_seqs.xlsx', index=False)
    if save_fa_each:
        for name, seq in muts_seqs:
            with open(f'{folder}/{name}.fa', 'w') as fa:
                fa.write(f'>{name}\n{seq}\n')
    if save_fa:
        with open(f'{folder}/mutants.fa', 'w') as fa:
            for name, seq in muts_seqs:
                fa.write(f'>{name}\n{seq}\n')
        
# read mut names to generate all other single mutations
def generate_all_other_single(seq_wt, seq_muts, save_df):
    seq_muts = set(seq_muts)
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    other_mut_seqs = []
    for pos in range(len(seq_wt)):
        for mut in aa:
            if mut != seq_wt[pos]:
                name = f'{seq_wt[pos]}{pos+1}{mut}'
                if name not in seq_muts:
                    other_mut = seq_wt[:pos] + mut + seq_wt[pos+1:]
                    other_mut_seqs.append([name, other_mut])
    df = pd.DataFrame(other_mut_seqs, columns=['mut', 'seq'])
    df.to_excel(save_df, index=False)

# drop NaN and assign fold index
def get_clean_data(data_file='', fold_idx=False):
    df = pd.read_excel(data_file, header=0, index_col=None, usecols=[0,1,2])
    df.dropna(axis=0, how='any', inplace=True)
    df = df.reset_index(drop=True)
    header = ['name','seq','yield']
    if fold_idx:
        df['mutation_type'] = df['name'].apply(lambda x: 'multi' if '-' in x else 'single')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        df['fold'] = -1
        for fold_number, (_, val_idx) in enumerate(skf.split(df, df['mutation_type'])):
            df.loc[val_idx, 'fold'] = fold_number
        df = df.drop('mutation_type', axis=1)
        header.append('fold')
    df.to_excel('./data.xlsx', index=False, header=header)

if __name__ == '__main__':
    file = ''
    seq_wt = str(SeqIO.read('', 'fasta').seq)
    seq_muts = pd.read_excel(file, header=0, usecols=[0]).values.ravel()
    short_to_seqs(seq_muts, seq_wt, save_fa=False, save_excel=True)
