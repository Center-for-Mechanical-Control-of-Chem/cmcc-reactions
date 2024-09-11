import abc, numpy as np

from rdkit import Chem
from . import util
from .reaction_graphs import ReactionGraph

__all__ = [
    "AtomMapper",
    "ChythonAtomMapper"
]

class AtomMapper(metaclass=abc.ABCMeta):
    def __init__(self, add_implicit_hydrogens=True, minimize_reaction_core=True):
        self.add_implicit_hydrogens = add_implicit_hydrogens
        self.minimize_reaction_core = minimize_reaction_core


    @abc.abstractmethod
    def map_atoms(self, reaction_smiles:str):
        ...

    def apply(self, smiles,
              add_hydrogens=None,
              minimize_reaction_core=None,
              check_mapping=True
              ):
        if add_hydrogens is None:
            add_hydrogens = self.add_implicit_hydrogens
        if minimize_reaction_core is None:
            minimize_reaction_core = self.minimize_reaction_core
        smiles = self.add_reaction_hydrogens(smiles,
                                             explicit_only=not add_hydrogens
                                             )

        remapped_smiles = self.map_atoms(smiles)
        if minimize_reaction_core:
            reactant_smiles, product_smiles = remapped_smiles.split(">>")
            reactant, prod = [util.parse_smiles(s) for s in [reactant_smiles, product_smiles]]
            (og, remapping), _ = ReactionGraph.get_core_minimizing_reindexing(reactant, prod)

            for i in og:
                reactant_smiles = reactant_smiles.replace(f":{i}]", f":<{i}>]")
            for i,j in zip(og, remapping):
                reactant_smiles = reactant_smiles.replace(f":<{i}>]", f":{j}]")
            remapped_smiles = reactant_smiles + ">>" + product_smiles

        if check_mapping:
            reactant_smiles, product_smiles = remapped_smiles.split(">>")
            reactant, prod = [util.parse_smiles(s) for s in [reactant_smiles, product_smiles]]
            new_reactant = util.parse_smiles(reactant_smiles)
            syms_1 = util.get_symbols(new_reactant)
            syms_2 = util.get_symbols(prod)
            ord_1 = np.argsort(util.get_atom_map(new_reactant))
            ord_2 = np.argsort(util.get_atom_map(prod))

            syms_1 = [syms_1[i] for i in ord_1]
            syms_2 = [syms_2[i] for i in ord_2]

            if any(s_1 != s_2 for s_1, s_2 in zip(syms_1, syms_2)):
                raise ValueError(f"initial atom mapping must at least align atom types {syms_1} != {syms_2}")

        return remapped_smiles


    @classmethod
    def add_reaction_hydrogens(cls, reaction_smiles, explicit_only=False):
        parts = reaction_smiles.split('>>')
        if len(parts) != 2:
            raise ValueError("Reaction SMILES requires products and reactants to be separated by >>")

        reactants, products = parts
        reactants = reactants.split('.')
        products = products.split('.')

        def add_hydrogens(smiles_list):
            mol_list = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES string: {smiles}")
                mol = Chem.AddHs(mol, explicitOnly=explicit_only)
                mol_list.append(Chem.MolToSmiles(mol))
            return mol_list

        reactants_with_h = add_hydrogens(reactants)
        products_with_h = add_hydrogens(products)

        reactants_smiles_with_h = '.'.join(reactants_with_h)
        products_smiles_with_h = '.'.join(products_with_h)
        return f"{reactants_smiles_with_h}>>{products_smiles_with_h}"

class ChythonAtomMapper(AtomMapper):
    def map_atoms(self, reaction_smiles):
        import chython
        r = chython.smiles(reaction_smiles)
        r.reset_mapping()
        return format(r, 'm')