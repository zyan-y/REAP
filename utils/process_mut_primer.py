import pandas as pd
import numpy as np
from Bio import SeqIO
import primer3


tm_prefer = [56, 60]
idx_pos_na = 5031
redundancy = 100

def estimate_tm(seq):
    return primer3.calc_tm(seq)

def reverse_complement(seq):
    seq = seq.upper()
    return seq.translate(str.maketrans('ATCG', 'TAGC'))[::-1]

def estimate_tm_rc(seq):
    return estimate_tm(reverse_complement(seq))


def design_one_mut(wt_seq, mut_short, codon_dict):
    wt, pos, mut = mut_short[0], int(mut_short[1:-1])-1, mut_short[-1]
    pos_na = pos*3 + redundancy
    if wt_seq[pos_na :pos_na+3 ] != codon_dict[wt]:
        print(f'May wrong codon at {pos_na}')
    mut_seq = wt_seq[0:pos_na] + codon_dict[mut] + wt_seq[pos_na+3:]
    
    primer_overlap = mut_seq[pos_na-10: pos_na+3+7]

    primer_F_num = 20
    primer_R_num = 20

    primer_F = mut_seq[pos_na+3: pos_na+3+primer_F_num]
    while estimate_tm(primer_F) < tm_prefer[0]:
        primer_F_num += 1
        primer_F = mut_seq[pos_na+3: pos_na+3+primer_F_num]
        
    while estimate_tm(primer_F) > tm_prefer[1]:
        primer_F_num -= 1
        primer_F = mut_seq[pos_na+3: pos_na+3+primer_F_num]
    
    primer_R = mut_seq[pos_na-primer_R_num :pos_na]
    while estimate_tm_rc(primer_R) < tm_prefer[0]:
        primer_R_num += 1
        primer_R = mut_seq[pos_na-primer_R_num :pos_na]
        
    while estimate_tm_rc(primer_R) > tm_prefer[1]:
        primer_R_num -= 1
        primer_R = mut_seq[pos_na-primer_R_num :pos_na]

    primer1 = primer_overlap + primer_F[7:]
    primer2 = reverse_complement(primer_R + mut_seq[pos_na:pos_na+3+7])
    
    data = [mut_short, pos_na+idx_pos_na, primer1, primer2, 
            primer_F, reverse_complement(primer_R), estimate_tm(primer_F), estimate_tm_rc(primer_R) ]
    return  data


def design_df_mut(muts, wt_seq, codon_pre_f='codon_preference.txt'):
    codon_pre = np.loadtxt(codon_pre_f, delimiter=':', dtype=str)
    codon_dict = {}
    for aa, codon in codon_pre:
        codon_dict[aa] = codon
    
    design_results = []
    for mut in muts:
        data = design_one_mut(wt_seq, mut, codon_dict)
        design_results.append(data)
        
    header = ['mut', 'mut start', 'primer_F', 'primer_R', 'primer_F_bind', 'primer_R_bind', 'tm_F', 'tm_R']
    df = pd.DataFrame(design_results, columns=header)
    df.to_excel('design_primers.xlsx', index=False)


if __name__ == '__main__':
    mut_df = ''
    muts = pd.read_excel(mut_df, sheet_name=0, header=0, usecols=[0]).astype(str).values.ravel()
    wt_seq = str(SeqIO.read('', 'fasta').seq)
    design_df_mut(muts, wt_seq)