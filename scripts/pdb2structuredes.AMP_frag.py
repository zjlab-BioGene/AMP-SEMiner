import os,sys,re
import numpy as np
import pandas as pd
import mdtraj as md
from prody import *
import networkx as nx
import community
import community as community_louvain
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import argparse
import logging
import shutil

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level='INFO')

## Parameters
ap = argparse.ArgumentParser()
ap.add_argument('--in_dir','-i',type=str,default=None,required=True,help='Input directory with .pdb files. required.')
ap.add_argument('--out_dir','-o',type=str,default='./',required=True,help='Output directory.')

def make_dirs(dir):
    try:
        os.makedirs(dir)
    except:
        print("Output dir already exists: %s" % dir)

def structural_des_cal(pdb):
    traj = md.load(pdb)
    
    # =========== 1.Basic structural descriptors ===========
    ## Calculate Radius of gyration (Rg in unit Angstrom)
    rg = round(md.compute_rg(traj)[0]*10.,6)
    # Calulate number of residues (N, chain length)
    n_resi = traj.n_residues
    rg_norm = rg/(n_resi**(1./3.))
    #Calculate solvent-accessible surface area (SASA in unit Angstrom square).
    sasa = md.shrake_rupley(traj)
    area = round(sum(sasa[0])*100.,6)
    area_per_resi = round(area/float(n_resi), 6)
    # Calculate the gyration tensor
    gy_tensor = md.compute_gyration_tensor(traj)[0]
    # Eigenvalues of the gyration tensor
    eigs = np.linalg.eigvalsh(gy_tensor)*100.
    # Length of the semi-axes
    Lx = round(eigs[0]**.5, 6)
    Ly = round(eigs[1]**.5, 6)
    Lz = round(eigs[2]**.5, 6)
    min_semi_axes = min(Lx,Ly,Lz)
    
    # =========== 2.Secondary structure ===========
    # Calculation of DSSP (Define Secondary Structure of Proteins)
    dssp1 = md.compute_dssp(traj, simplified=False)[0]
    dssp2 = md.compute_dssp(traj, simplified=True)[0]
    # Calculate the loop/coil fraction
    n_loop = 0
    n_coil = 0
    for w in dssp1:
        if w == ' ':
            n_loop += 1
    for w in dssp2:
        if w == 'C':
            n_coil += 1          
    frac_loop = round(float(n_loop)/float(len(dssp1)), 6)
    frac_coil = round(float(n_coil)/float(len(dssp2)), 6)
    
    # =========== 3.Topologic descriptors ===========
    # calculation of topological descriptors
    try:
        pdb_full = parsePDB(pdb)
        calphas = pdb_full.select('calpha')
        N = len(calphas)
        # Cutoff distance
        rc = 8.
        # Build Kirchhoff matrix by Prody (graph Laplacain)
        gnm = GNM('GNM_'+ pdb)
        gnm.buildKirchhoff(calphas, cutoff=rc)
        tmpHessian = gnm.getKirchhoff()
        # Build the network by NetworkX
        G = nx.Graph()
        for i in range(N):
            # Add nodes
            G.add_node(i)
        for i in range(N):
            for j in range(i+1, N):
                if j>i and tmpHessian[i][j]<-.9:
                    # Add edges
                    G.add_edge(i,j)          
        # Calculate Assortativity
        assort = nx.degree_assortativity_coefficient(G)
        # Find the partition of the network
        parti = community_louvain.best_partition(G)
        # Calculate modularity
        Q = community.modularity(parti, G)
    except:
        assort,parti,Q = 0,0,0
    # =========== 4.Fractal dimension (based on box-counting method) ===========
    try:
        rclist = np.array([5, 6, 7, 8, 9, 10, 11, 12])
        nb_all = []
        Dist = buildDistMatrix(calphas)
        # Changing the cutoff distances
        for rc in rclist:
            nb = []
            for i in range(len(Dist)):
                # Calcualting the number of neighbors
                pos_count = len(list(filter(lambda x: (x <= rc), Dist[i])))
                nb.append(pos_count)
            nb_all.append(nb)
        nb_all = np.array(nb_all)
        nb_ave = np.array([round(np.mean(nb_all[k]),4) for k in range(len(nb_all))])
        # Scattering plot
        plt.plot(rclist, nb_ave, 'ro')
        plt.xlabel('$r$')
        plt.ylabel('Average number of neighbors')
        # Log-log fitting
        kb = np.polyfit(np.log(rclist), np.log(nb_ave), 1)
        fitted = [np.exp(kb[0]*np.log(x)+kb[1]) for x in rclist]
        plt.plot(rclist, fitted, 'k-')
        # Fractal dimension
        fract = round(kb[0], 6)
    except:
        fract = 0
    
    # =========== 5.Sequence (Hydrophobic-hydrophilic segregation) ===========
    try:
        seq = calphas.getSequence()
        analyzed_seq = ProteinAnalysis(seq)
        # Set moving window size
        iwindow = 7
        # Zimmerman hydrophobicity scale
        zi = {"A": 0.83, "R": 0.83, "N": 0.09, "D": 0.64, "C": 1.48,
            "Q": 0, "E": 0.65, "G": 0.1, "H": 1.1, "I": 3.07,
            "L": 2.52, "K": 1.6, "M": 1.4, "F": 2.75, "P": 2.7,
            "S": 0.14, "T": 0.54, "W": 0.31, "Y": 2.97, "V": 1.79}
        # Calculate and plot the hydrophathy profile of the protein 
        scale_HP = analyzed_seq.protein_scale(window=iwindow, param_dict=zi)
        plt.plot(scale_HP)
        plt.xlabel('Residue index')
        plt.ylabel('Hydrophobicity (Zimmerman)')
        # Calculate CV_HP
        ave = np.mean(scale_HP)
        stdev = np.std(scale_HP)
        CVHP = stdev/ave
    except:
        CVHP = 0
        
    return [rg,rg_norm,area,area_per_resi,min_semi_axes,frac_loop,frac_coil,assort,Q,fract,CVHP]
       
            
def get_structural_descriptors(input_dir,outdir):
    col_names = ['ProteinID',
                 'Rg','RgNorm','Area','AreaPerResi','MinSemiAxes','Loop','Coil','Assort','Q','Dim','CVHP',]
    data = pd.DataFrame(columns=col_names)
    
    for pdb in os.listdir(input_dir):
        if pdb.endswith('pdb'):
            protein_pdb = os.path.join(input_dir,pdb)
            name = re.sub('\.pdb$','',pdb)
            try:
                protein_metrics = structural_des_cal(protein_pdb)
                data.loc[len(data.index)] = [name,] + protein_metrics
            except:
                print('Failed at %s' % pdb)
    data.to_csv(os.path.join(outdir,'Structural_descriptors.txt'),sep='\t',index=False)

'''
Descriptors:
1. rg. Radius of gyration (Rg in unit Angstrom).
2. rg_norm. Normalied rg.
3. area. Solvent-accessible surface area.
4. area_per_resi. 
5. min_semi_axes. The shortest of the length of the semi-axes (Lx, Ly, Lz).
6. frac_loop. Loop fraction (based on unsimplifed DSSP).
7. frac_coil. Coil fraction (based on simplifed DSSP).
8. assort. Assortativity.
9. Q. Modularity.
10. fract. Fractal dimension (based on box-counting method).
11. CVHP. Sequence (Hydrophobic-hydrophilic segregation).
'''

def main():
    args = ap.parse_args()
    make_dirs(args.out_dir)
    get_structural_descriptors(args.in_dir,args.out_dir)
     
if __name__ == '__main__':
    main()
