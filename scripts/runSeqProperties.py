#!/home/lwh/miniconda3/envs/pepfun-env/bin/python
import sys
sys.path.append('/home/lwh/00.data/Tools/PepFun')

import os, re
import pandas as pd
import numpy as np
import argparse
from pepfun import *
from modlamp.descriptors import GlobalDescriptor

hydrophobic_moment_script = '/home/lwh/00.data/Tools/hydrophobic_moment/hydrophobic_moment.py'

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='Run pepFun and molAMP.')
    parser.add_argument('--input_tab', '-i', type=str, default=None, required=True, help='Input AMP-prediction table. (e.g. ./test_data.pred.csv)')
    parser.add_argument('--out_prefix', '-o', type=str, default=None, required=True, help='Input AMP-prediction table. (e.g. ./test_data.AMP)')
    parser.add_argument('--max_len', type=int, default=30, help='Maximum-length of AMP candidates (default 30).')
    parser.add_argument('--min_len', type=int, default=15, help='Minimum-length of AMP candidates (default 15).')
    args = parser.parse_args()
    if (args.input_tab==None) or (args.out_prefix==None):
        raise Exception('Error: input AMP-candidate table and output prefix are required!')
    
    return args

## Run pepFun & molAMP
def run_pepFun_molAMP(input_tab, out_prefix, max_len, min_len):
    
    # indf = pd.read_csv(input_tab, sep='\t',header=None)
    # indf.columns = ['dbName', 'ProID', 'AMP', 'AMPlen', 'Position', 'Sequence']
    indf = pd.read_csv(input_tab, sep='\t')
    indf = indf[(indf.AMPlen>=min_len) & (indf.AMPlen<=max_len)]
    outdf = pd.DataFrame(columns=['TMPID','netCharge','molWeight','avgHydro','isoelectricPoint','BomanIndex',
                                  'Solubility_rules_failed','Synthesis_rules_failed', 'CrippenLogP',
                                  'maxHydrophobicMoment','meanHydrophobicMoment','HydrophobicMoment',])
    with open(out_prefix+'.amp.faa', 'w') as handle:
        sequences = indf.AMP.tolist()
        names = indf.TMPID.tolist()

        for i in range(len(sequences)):
            seq = sequences[i]
            name = names[i]
            ## pepFun
            pep=peptide_sequence(seq)
            pep.compute_peptide_charges()
            pep.calculate_properties_from_mol()
            pep.calculate_properties_from_sequence()
            pep.solubility_rules()
            pep.synthesis_rules()
            ## molAMP
            desc = GlobalDescriptor(seq)
            desc.boman_index()
            ## save to DataFrame
            outdf.loc[len(outdf.index)] = [ name, pep.netCharge, pep.mol_weight, pep.avg_hydro, pep.isoelectric_point, desc.descriptor[0][0],
                                            pep.solubility_rules_failed, pep.synthesis_rules_failed, pep.mol_logp, 
                                            None, None, None, ]
            ## output to .faa   
            handle.write('>'+name+'\n'+seq+'\n')

    return indf, outdf, out_prefix+'.amp.faa'

def run_HydrophobicMoment(input_faa, out_prefix):
    ## run hydrophobic_moment.py script
    out_tab_1 = out_prefix + '.hydmom.out1.csv'
    out_tab_2 = out_prefix + '.hydmom.out2.csv'
    os.system('python %s -f %s -o %s -w 18' % (hydrophobic_moment_script, input_faa, out_tab_1))
    os.system('python %s -f %s -o %s -w 300' % (hydrophobic_moment_script, input_faa, out_tab_2))
    
    maxhydmom, meanhydmon = {}, {}
    dataframe_1 = pd.read_csv(out_tab_1)
    for name, df in dataframe_1.groupby(by='Name'):
        maxhydmom[name] = df['Mean Hydrophobic Moment'].max()
        meanhydmon[name] = df['Mean Hydrophobic Moment'].mean()
    
    dataframe_2 = pd.read_csv(out_tab_2)
    hydmon = dict(zip(dataframe_2['Name'].tolist(), dataframe_2['Mean Hydrophobic Moment'].tolist()))
    
    os.remove(out_tab_1)
    os.remove(out_tab_2)
    os.remove(input_faa)
    
    return maxhydmom, meanhydmon, hydmon

def merge_datatable(indf, outdf, maxhydmom, meanhydmon, hydmon, out_prefix):
    outdf['maxHydrophobicMoment'] = [ maxhydmom[x] for x in outdf['TMPID'].tolist() ]
    outdf['meanHydrophobicMoment'] = [ meanhydmon[x] for x in outdf['TMPID'].tolist() ]
    outdf['HydrophobicMoment'] = [ hydmon[x] for x in outdf['TMPID'].tolist() ]
    outdf = pd.merge(indf,outdf, left_on='TMPID',right_on='TMPID',how='left')
    outdf.to_csv(out_prefix+'.SeqProperties.txt', sep='\t', index=False)
    
if __name__ == '__main__':
    args = get_parameters()
    indf, outdf, out_faa = run_pepFun_molAMP(args.input_tab, args.out_prefix, args.max_len, args.min_len)
    maxhydmom, meanhydmon, hydmon = run_HydrophobicMoment(out_faa, args.out_prefix)
    merge_datatable(indf, outdf, maxhydmom, meanhydmon, hydmon, args.out_prefix)
    
    
## Run: python runSeqProperties.py -i test_data.pred.csv -o ./test_out    
               

