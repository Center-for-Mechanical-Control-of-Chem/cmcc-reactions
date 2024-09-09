
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

    def __init__(self, reactants, products):
        self.reactant_set = reactants
        self.product_set = products
        self._unified_reactant = None
        self._unified_product = None

    @property
    def reactant(self):
        ...

    @classmethod
    def combine_mols(cls,
                     reactant_mols,
                     energy_evaluator=None,
                     model=None, charge=None,
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
        all_reactant_coms = [
            mol.mol.get_center_of_mass()
            for mol in reactant_mols
        ]
        com_shifted_structs = [
            struct - com[np.newaxis]
            for struct, com in zip(all_reactant_structs, all_reactant_coms)
        ]
        all_charges = [
            mol
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
            total_disp += rad  # pad to avoid next neighbor

        full_struct = np.concatenate(final_structs, axis=0)
        return GenericMol(

        )
        mol = Atoms(symbols=sum([mol.symbols for mol in reactant_mols], []), positions=full_struct)
        mol.calc = setup_AIMNET(model, charge)

        return mol