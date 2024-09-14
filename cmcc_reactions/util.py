
import numpy as np
from ase.visualize import view
from rdkit import Chem

from .conversions import *
from .mol_types import *

__all__ = ["sort_mol", "view_mol", "export_sdf",  "export_traj",
           "parse_smiles",
           "get_symbols",
           "get_positions",
           "get_charge",
           "get_bonds",
           "get_atom_map",
           "get_submol",
           "determine_bonds"
           ]

def sort_mol(mol):
    generic_mol = convert(mol, GenericMol)  # type: GenericMol

    sym = generic_mol.symbols
    pos = generic_mol.positions
    charges = generic_mol.charges
    atom_map = generic_mol.atom_map
    bonds = generic_mol.bonds

    nat = len(atom_map)
    atom_map_sorting = [
        nat if map_num == 0 else map_num
        for map_num in atom_map
    ]
    ord = np.argsort(atom_map_sorting)
    inv = np.argsort(ord)

    pos = [pos[i] for i in ord]
    sym = [sym[i] for i in ord]
    charges = [charges[i] for i in ord]
    atom_map = [atom_map[i] for i in ord]

    bonds = [
        [inv[b[0]], inv[b[1]], b[2]]
        for b in bonds
    ]

    bond_ord = np.lexsort(
        [
            [b[1] for b in bonds],
            [b[0] for b in bonds]
        ]
    )
    bonds = [bonds[i] for i in bond_ord]

    return GenericMol(
        sym,
        pos,
        charges,
        bonds=bonds,
        atom_map=atom_map
    )

def view_mol(mol, viewer='x3d'):
    return view(convert(mol, ASEMol).mol, viewer=viewer)

def determine_bonds(mol):
    from rdkit.Chem import rdDetermineBonds

    conf = convert(mol, Chem.Conformer)
    rdDetermineBonds.DetermineConnectivity(conf)
    rdDetermineBonds.DetermineBondOrders(conf, charge=np.sum(get_charge(mol)))

    return convert(conf, type(mol))

def export_sdf(mol, file, guess_bonds=False):
    if guess_bonds:
        mol = determine_bonds(mol)
    sdf = convert(mol, SDFString).string
    with open(file, 'w+') as dump:
        dump.write(sdf)

def export_traj(traj, file, guess_bonds=False):
    if guess_bonds:
        traj = [determine_bonds(mol) for mol in traj]
    sdf = "\n\n$$$$\n\n".join(convert(mol, SDFString).string for mol in traj)
    with open(file, 'w+') as dump:
        dump.write(sdf)

def get_symbols(mol):
    if isinstance(mol, Chem.Mol):
        return [atom.GetSymbol() for atom in mol.GetAtoms()]
    else:
        return convert(mol, GenericMol).symbols

def get_positions(mol):
    return convert(mol, GenericMol).positions

def get_bonds(mol):
    if isinstance(mol, Chem.Mol):
        return [
            [b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()]
            for b in mol.GetBonds()
        ]
    else:
        return convert(mol, GenericMol).bonds

def get_charge(mol):
    if isinstance(mol, Chem.Mol):
        return [atom.GetCharge() for atom in mol.GetAtoms()]
    else:
        return convert(mol, GenericMol).charge

def get_atom_map(mol):
    if isinstance(mol, Chem.Mol):
        return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    else:
        return convert(mol, GenericMol).atom_map

def get_submol(mol, inds):
    sym = get_symbols(mol)
    pos = get_positions(mol)
    charge = get_charge(mol)
    amap = get_atom_map(mol)
    bonds = get_bonds(mol)
    set_inds = set(inds)
    return GenericMol(
        [sym[i] for i in inds],
        [pos[i] for i in inds],
        [charge[i] for i in inds],
        bonds=[
            b
            for b in bonds
            if b[0] in set_inds or b[1] in set_inds
        ],
        atom_map=[amap[i] for i in inds]
    )

def parse_smiles(smiles, add_implicit_hydrogens=True):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    rdkit_mol = Chem.MolFromSmiles(smiles, params)
    return Chem.AddHs(rdkit_mol, explicitOnly=not add_implicit_hydrogens)