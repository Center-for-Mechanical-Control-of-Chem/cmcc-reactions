
import numpy as np, networkx, itertools
import McUtils.Numputils as nput

from . import util

__all__ = [
    "ReactionGraph"
]

class ReactionGraph:
    def __init__(self, reactants, products):
        self.reactants = reactants
        self.products = products
        self._core = None
        # self.graph1 = self.get_mol_graph(reactants)
        # self.graph2 = self.get_mol_graph(products)

    @property
    def reactive_core(self):
        if self._core is None:
            self._core = self.get_reacting_atoms(self.reactants, self.products)
        return self._core

    @classmethod
    def get_bond_set(cls, mol):
        bonds_1 = util.get_bonds(mol)
        return { tuple(sorted([i,j])):t for i,j,t in bonds_1 }
    @classmethod
    def get_reacting_atoms(cls, mol_1, mol_2):
        bond_set_1 = cls.get_bond_set(mol_1)
        bond_set_2 = cls.get_bond_set(mol_2)

        changes = (bond_set_1.keys() - bond_set_2.keys() )
        return np.unique(np.array(list(changes)))

    @classmethod
    def get_core_minimizing_reindexing(cls, mol_1, mol_2):
        syms_1 = util.get_symbols(mol_1)
        syms_2 = util.get_symbols(mol_2)
        map_1 = util.get_atom_map(mol_1)
        ord_1 = np.argsort(map_1)
        ord_2 = np.argsort(util.get_atom_map(mol_2))

        syms_1 = [syms_1[i] for i in ord_1]
        syms_2 = [syms_2[i] for i in ord_2]

        if any(s_1 != s_2 for s_1, s_2 in zip(syms_1, syms_2)):
            raise ValueError(f"initial atom mapping must at least align atom types {syms_1} != {syms_2}")
        # we start with initial core to reduce work done
        inv_1 = np.argsort(ord_1)
        inv_2 = np.argsort(ord_2)
        bond_set_1 = {
            (inv_1[i], inv_1[j])
            for i,j in cls.get_bond_set(mol_1).keys()
        }
        bond_set_2 = {
            (inv_2[i], inv_2[j])
            for i, j in cls.get_bond_set(mol_2).keys()
        }

        # initial reactant core
        test_bonds = np.unique(np.array(
            list(bond_set_1 - bond_set_2)
            + list(bond_set_2 - bond_set_1)
        ))
        # permutable_bonds = {
        #     (i, j)
        #     for n, (i, j,) in enumerate(bond_set_1)
        #     if i in test_bonds or j in test_bonds
        # }

        # permutable groups
        sym_splits, _ = nput.group_by(np.arange(len(syms_1)), np.array([ord(s) for s in syms_1]))
        perm_blocks = []
        perm_atoms = []
        for _, atom_inds in zip(*sym_splits):
            atom_inds = nput.intersection(atom_inds, test_bonds)[0]  # only permute things in the original core
            if len(atom_inds) > 0:
                perm_atoms.append(atom_inds)
                perm_blocks.append(itertools.permutations(atom_inds))

        nsym = len(syms_1)
        core_size = len(test_bonds)
        perm = np.arange(nsym)
        for full_perm in itertools.product(*perm_blocks):
            reindexing = np.arange(nsym)
            for atom_inds, new_idx in zip(perm_atoms, full_perm):
                reindexing[atom_inds,] = new_idx
            new_bond_set_1 = {
                (reindexing[i], reindexing[j])
                for (i, j) in bond_set_1
            }
            # print(reindexing)
            # print(full_perm)
            new_core = np.unique(np.array(
                list(new_bond_set_1 - bond_set_2)
                + list(bond_set_2 - new_bond_set_1)
            ))
            # print(len(new_core))
            if len(new_core) < core_size:
                perm = reindexing

        # perm = inv_1[perm]
        return (map_1, [map_1[i] for i in perm]), (perm, core_size)

    @classmethod
    def get_mol_graph(cls, mol) -> networkx.Graph:
        bonds = util.get_bonds(mol)
        sym = util.get_symbols(mol)
        graph = np.zeros((len(sym),len(sym)))
        for i, j, _ in bonds:
            graph[i, j] = graph[j, i] = 1
        return networkx.from_numpy_array(graph)

    # def get_reaction_core(self):
    #     return networkx.optimal_edit_paths(self.graph1, self.graph2)
