
import numpy as np

import ase
from ase.build import minimize_rotation_and_translation
from rdkit import Chem
from rdkit.Chem import AllChem

from .energy_evaluators import *
from .conversions import *
from .util import *
from .mol_types import *

__all__ = [
    "ReactionComponent",
    "ReactionComponentSet"
]

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
        params = Chem.SmilesParserParams()
        params.removeHs = False
        rdkit_mol = Chem.MolFromSmiles(smiles, params)
        rdkit_mol = Chem.AddHs(rdkit_mol, explicitOnly=not add_implicit_hydrogens)
        # rdkit_mol = Chem.AddHs(rdkit_mol, explicitOnly=not add_implicit_hydrogens)

        if mmff_params is None:
            mmff_params = MMFFEnergyEvaluator.get_default_params()

        if energy_evaluator is None:
            num_conformers = 1

        conformer_set = AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=num_conformers, params=mmff_params)
        conformers = [
            rdkit_mol.GetConformer(conf_id)
            for conf_id in conformer_set
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

    def __init__(self,
                 components:'list[ReactionComponent]',
                 unified=None,
                 **kw
                 ):
        self.component_set = components
        if unified is None:
            self.unified_component = self.combine_mols([r.mol for r in components], **kw)

    @classmethod
    def from_smiles(cls, smiles,
                    num_conformers=1,
                    energy_evaluator: EnergyEvaluator = None,
                    optimize=False,
                    add_implicit_hydrogens=True,
                    mmff_params=None,
                    reference_structure=None
                    ):
        component_smiles = smiles.split(".")
        components = [
            ReactionComponent.from_smiles(smiles,
                                          num_conformers=num_conformers,
                                          energy_evaluator=energy_evaluator,
                                          optimize=optimize,
                                          add_implicit_hydrogens=add_implicit_hydrogens,
                                          mmff_params=mmff_params
                                          )
            for smiles in component_smiles
        ]
        return cls(components,
                   reference_structure=reference_structure,
                   energy_evaluator=energy_evaluator
                   )

    @classmethod
    def combine_mols(cls,
                     reactant_mols,
                     displacement_scaling=1.2,
                     displacement_direction=(1, 0, 0),
                     reference_structure=None,
                     alignment_method="bond_tightening",
                     energy_evaluator=None
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
            mol.charges
            for mol in ase_mols
        ]
        all_syms = [
            mol.mol.symbols
            for mol in ase_mols
        ]
        all_bonds = [
            mol.bonds
            for mol in ase_mols
        ]
        all_reactant_coms = [
            mol.mol.get_center_of_mass()
            for mol in ase_mols
        ]
        com_shifted_structs = [
            struct - com[np.newaxis]
            for struct, com in zip(all_reactant_structs, all_reactant_coms)
        ]
        all_atom_maps = [
            mol.atom_map
            for mol in ase_mols
        ]
        # diplace each mol so that it's shifted from the others by more than its molecular radius
        molecular_radii = [
            displacement_scaling * np.max(np.linalg.norm(struct, axis=1))
            for struct in com_shifted_structs
        ]

        full_syms = sum([list(syms) for syms in all_syms], [])
        full_charges = np.concatenate(all_charges)
        full_atom_amp = sum([list(atom_map) for atom_map in all_atom_maps], [])

        full_bonds = []
        num_shift = 0
        for struct, bonds in zip(com_shifted_structs, all_bonds):
            new_bonds = [
                [b[0] + num_shift, b[1] + num_shift, b[2]]
                for b in bonds
            ]
            full_bonds.append(new_bonds)
            num_shift += len(struct)
        full_bonds = sum(full_bonds, [])

        if reference_structure is not None:
            mapped_structs = []
            mapped_coms = []
            reference_structure = convert(reference_structure, ASEMol)# type:ASEMol
            # align each subset
            inv_map = np.argsort(np.argsort(full_atom_amp))  # regenerate original from final
            d = 0
            ref_pos = reference_structure.mol.positions
            for init_struct,mol in zip(com_shifted_structs, ase_mols):
                mol = mol #type:ASEMol
                map = inv_map[d:d + len(mol.mol.symbols)]
                submap = np.full(len(inv_map), 1000, dtype=int)
                submap[map,] = np.arange(len(map))
                ref_struct = ase.Atoms(
                    symbols=mol.mol.symbols,
                    positions=ref_pos[map,]
                )

                if alignment_method == "bond_tightening":
                    # all bonds get mapped back to their original lengths]
                    pos = ref_struct.get_positions().copy()
                    bonds = get_bonds(mol)
                    # bonds = [
                    #     (submap[i], submap[j])
                    #     for i,j,_ in bonds
                    # ]

                    og_bond_lengths = [
                        np.linalg.norm(init_struct[i] - init_struct[j])
                        for i,j,_ in bonds
                    ]
                    for iteration in range(50):
                        for (i,j,_),l in zip(bonds, og_bond_lengths):
                            new_vec = pos[i] - pos[j]
                            new_len = np.linalg.norm(new_vec)
                            center = (pos[i] + pos[j]) / 2
                            disp = l/2 * (new_vec/new_len)
                            pos[i] = center + disp
                            pos[j] = center - disp

                    min_struct = ase.Atoms(mol.mol.symbols, pos)
                elif alignment_method == "rmsd":
                    min_struct = ase.Atoms(mol.mol.symbols, init_struct)
                else:
                    raise ValueError(f"unknown alignment method '{alignment_method}'")

                minimize_rotation_and_translation(ref_struct, min_struct)#type:ase.Atoms
                if energy_evaluator is not None:
                    min_mol = ASEMol(min_struct, mol.charges, mol.bonds, mol.atom_map)
                    min_mol = energy_evaluator.optimize_structure(min_mol)
                    min_struct = min_mol.mol
                mapped_structs.append(min_struct.positions)
                mapped_coms.append(min_struct.get_center_of_mass())
                d += len(init_struct)
            full_struct = np.concatenate(mapped_structs, axis=0)
            full_mol = ase.Atoms(full_syms, full_struct)
            full_com = full_mol.get_center_of_mass()

            final_structs = []
            for struct, com in zip(mapped_structs, mapped_coms):
                com_disp = com - full_com
                final_structs.append(
                    struct + displacement_scaling * com_disp[np.newaxis, :]
                )

        else:
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

        return sort_mol(GenericMol(
            full_syms,
            full_struct,
            charges=full_charges,
            bonds=full_bonds,
            atom_map=full_atom_amp
        ))