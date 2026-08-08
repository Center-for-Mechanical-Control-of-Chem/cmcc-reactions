import itertools
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
from ase import Atoms
from aimnet2calc import AIMNet2ASE
from ase.optimize import BFGS
from ase.io import write

def generate_smiles(template, R):
    smiles = template.replace("[R]", R)
    return smiles

def gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0)
    return list(ids)

R_list = ["O=S(=O)(C2=CC=CC=C2)", "NC", "NC(=O)", "FC(C=C3)=CC=C3"]
R_heavy_atom_indices = [9, 2, 3, 7]

template_1 = "[R]C1(C)C(O2)(C3=CC=CC=C3)C(C=CC=C4)=C4C2(C5=CC=CC=C5)C1"


All_smiles = [] 
Indices = []
for idx in range(len(R_list)):
    smiles = generate_smiles(template_1, R_list[idx])
    idx1, idx2 = R_heavy_atom_indices[idx] + 1, R_heavy_atom_indices[idx] + 1 + 1 + 1
    idx3, idx4 = R_heavy_atom_indices[idx] + 1 + 1 + 15, R_heavy_atom_indices[idx] + 1 + 1 + 22
    All_smiles.append(smiles)
    Indices.append([[idx1,idx2],[idx3,idx4]])

Canonical_smiles = []
for i in All_smiles:
    Canonical_smiles.append(Chem.CanonSmiles(i))
uniq_smiles, uniq_indices = np.unique(Canonical_smiles,return_index=True)

f = open('R16_smiles.txt', 'w')
for i in range(len(uniq_indices)):
    tmp_smiles = All_smiles[uniq_indices[i]]
    f.write(tmp_smiles + '\n')
f.close()

calc = AIMNet2ASE('aimnet2',charge=0)
for i in range(len(uniq_indices)):
    tmp_smiles = All_smiles[uniq_indices[i]]
    mol = Chem.MolFromSmiles(tmp_smiles)
    mol = Chem.AddHs(mol)

    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
    Energy_all = []
    for j in gen_conformers(mol):
        conf = mol.GetConformer(j)
        coord = conf.GetPositions()
        conformer_tmp = Atoms(symbols=atoms, positions=coord)
        conformer_tmp.calc = calc
        energy = conformer_tmp.get_potential_energy()
        Energy_all.append(energy[0])

    Energy_all = np.array(Energy_all)
    Energy_relative = Energy_all - min(Energy_all)
    if len(Energy_relative) <= 10:
        top_indices = [x for x in range(len(Energy_relative))]
    else:
        top_indices = np.argpartition(Energy_relative, 10)[:10]

    try:
        os.mkdir('Reaction16/'+str(i))
    except:
        pass

    for j in top_indices:
        conf = mol.GetConformer(int(j))
        coord = conf.GetPositions()
        conformer_tmp = Atoms(symbols=atoms, positions=coord)
        conformer_tmp.calc = calc
        opt = BFGS(conformer_tmp, logfile=None)
        opt.run(fmax=0.000027)
        try:
            os.mkdir('Reaction16/'+str(i)+'/'+str(j))
        except:
            pass

        write('Reaction16/'+str(i)+'/'+str(j)+'/conf.xyz', conformer_tmp, format='xyz')

        f = open('Reaction16/'+str(i)+'/'+str(j)+'/isomer.txt','w')
        f.write('BREAK ' + str(Indices[uniq_indices[i]][0][0]) + " " + str(Indices[uniq_indices[i]][0][1]) + '\n')
        f.write('BREAK ' + str(Indices[uniq_indices[i]][1][0]) + " " + str(Indices[uniq_indices[i]][1][1]) + '\n')
        f.close()
