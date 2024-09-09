
import numpy as np

import ase
from rdkit import Chem
from rdkit.Chem import AllChem

from .energy_evaluators import *
from .conversions import *
from .mol_types import *

class ReactionComponent:
    """
    A wrapper for handling reactants that uses `rdkit` and `ase`
    to implement chemical transformations
    """
    def __init__(self, mol):
        self.mol = mol

    @classmethod
    def from_smiles(cls, smiles,
                    num_conformers=1,
                    energy_evaluator:EnergyEvaluator=None,
                    optimize=False,
                    add_implicit_hydrogens=True,
                    mmff_params=None
                    ):
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if add_implicit_hydrogens:
            rdkit_mol = Chem.AddHs(rdkit_mol)

        if mmff_params is None:
            mmff_params = MMFFEnergyEvaluator.get_default_params()

        if energy_evaluator is None:
            num_conformers = 1

        conformer_set = AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=1, params=mmff_params)
        conformers = [
            conformer_set.GetConformer(conf_id)
            for conf_id in range(num_conformers)
        ]
        if energy_evaluator is not None:
            if optimize:
                for i, conf in enumerate(conformers):
                    conformers[i] = energy_evaluator.optimize_structure(conf)
            energies = [
                energy_evaluator.calculate_energy(mol)
                for mol in conformers
            ]
            min_e_conf = np.argmin(energies)
        else:
            min_e_conf = 0

        return cls(conformers[min_e_conf])

class ReactionComponentSet:

    def __init__(self, reactants, unified=None):
        self.reactant_set = reactants
        if unified is None:
            self.unified_reactant = self.combine_mols(reactants)

    @classmethod
    def combine_mols(cls,
                     reactant_mols,
                     displacement_scaling=2,
                     displacement_direction=(1, 0, 0)
                     ):
        ase_mols = [
            convert(mol, ASEMol)
            for mol in reactant_mols
        ] # type: list[ASEMol]
        all_reactant_structs = [
            mol.mol.get_positions()
            for mol in ase_mols
        ]
        all_charges = [
            mol.mol.get_charges()
            for mol in reactant_mols
        ]
        all_syms = [
            mol.mol.symbols
            for mol in reactant_mols
        ]
        all_bonds = [
            mol.bonds
            for mol in reactant_mols
        ]
        all_reactant_coms = [
            mol.mol.get_center_of_mass()
            for mol in reactant_mols
        ]
        com_shifted_structs = [
            struct - com[np.newaxis]
            for struct, com in zip(all_reactant_structs, all_reactant_coms)
        ]
        # diplace each mol so that it's shifted from the others by more than its molecular radius
        molecular_radii = [
            displacement_scaling * np.max(np.linalg.norm(struct, axis=1))
            for struct in com_shifted_structs
        ]

        final_structs = []
        total_disp = 0
        displacement_direction = np.asanyarray(displacement_direction)
        for i, (rad, struct) in enumerate(zip(molecular_radii, com_shifted_structs)):
            if i > 0:
                total_disp += rad  # add my radius
            struct = struct + displacement_direction * total_disp
            final_structs.append(struct)
            total_disp += rad  # pad to avoid next neighbor
        full_struct = np.concatenate(final_structs, axis=0)

        full_bonds = []
        num_shift = 0
        for struct, bonds in zip(com_shifted_structs, all_bonds):
            new_bonds = [
                [b[0]+num_shift, b[1]+num_shift, b[2]]
                for b in bonds
            ]
            full_bonds.append(new_bonds)
            num_shift += len(struct)
        full_bonds = sum(full_bonds, [])

        full_syms = sum([list(syms) for syms in all_syms], [])
        full_charges = np.concatenate(all_charges)

        return GenericMol(
            full_syms,
            full_struct,
            charges=full_charges,
            bonds=full_bonds
        )