
import collections, functools, numpy as np, tempfile as tf, os

import ase
from rdkit import Chem
from .mol_types import *

__all__ = [
    "convert",
    "register",
    "converters"
]

converters = {}
def register(from_type, to_type, conversion=None):
    if conversion is None:
        @functools.wraps(register)
        def register_conversion(conv):
            register(from_type, to_type, conv)
        return register_conversion
    else:
        if from_type not in converters: converters[from_type] = {}
        converters[from_type][to_type] = conversion

def bfs_search(converter_graph, from_type, target_type):
    bfs_queue = collections.deque()
    bfs_queue.append([[from_type]])
    visited = set()

    while bfs_queue:
        path, keys = bfs_queue.popleft()
        from_type = path[-1]
        visited.add(from_type)
        subgraph = converter_graph[from_type]
        for to_type in subgraph.keys() - visited:
            subpath = path + [to_type]
            if to_type == target_type:
                return subpath
            else:
                bfs_queue.append(subpath)
    else:
        raise ValueError(f"path from {from_type} to {target_type} not found")

def find_conversion(from_type, to_type):
    converter_path = bfs_search(converters, from_type, to_type)
    funcs = []
    prev = from_type
    for new_type in converter_path[1:]:
        funcs.append(converters[prev][new_type])
        prev = new_type
    def convert(mol, _converters=funcs):
        for c in _converters:
            mol = c(mol)
        return mol

    return convert

def convert(mol, target_type:type) -> 'target_type':
    from_type = type(mol).__qualname__
    to_type = target_type.__qualname__

    if from_type == to_type:
        return mol

    converter = find_conversion(from_type, to_type)
    return converter(mol)


@register(Chem.Conformer, GenericMol)
def rdkit_to_aps(conf:Chem.Conformer) -> GenericMol:
    mol = conf.GetOwningMol()

    sym = [
        atom.GetSymbol()
        for atom in mol.GetAtoms()
    ]
    pos = conf.GetPositions()
    charges = [
        atom.GetFormalCharge()
        for atom in mol.GetAtoms()
    ]
    bonds = [
        [b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()]
        for b in mol.GetBonds()
    ]

    atom_map_sorting = [
        atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
    ]
    ord = np.argsort(atom_map_sorting)
    inv = np.argsort(ord)

    pos = [pos[i] for i in ord]
    sym = [sym[i] for i in ord]
    charges = [charges[i] for i in ord]

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
        symbols=sym,
        positions=pos,
        charges=charges,
        bonds=bonds
    )

def _format_counts_line(npos, nbond):
    return "{:3.0f}{:3.0f}{:3.0f}     0  0  0  0  0  0999 V2000".format(npos, nbond, 0)
@register(GenericMol, SDFString)
def mol_to_str(mol:GenericMol):
    block = []
    block.append("MOL_NAME") # tbd
    block.append("     MOL NOTE") # tbd
    block.append("")

    npos = len(mol.positions)
    nbond = len(mol.bonds)

    block.append(_format_counts_line(npos, nbond))
    for (x,y,z), sym in zip(
        mol.positions,
        mol.symbols
    ):
        block.append(
            " {:10.4f}{:10.4f}{:10.4f} {:<2}  0  0  0  0  0  0  0  0  0  0  0  0".format(
                x, y, z, sym
            )
        )
    for i,j,t in mol.bonds:
        block.append("{:3}{:3}{:3}  0  0  0  0".format(i,j,t))
    for i,charge in enumerate(mol.charges):
        if charge != 0:
            block.append(" M CHG 1 {} {}".format(i+1, charge))
    block.append("M  END")
    return "\n".join(block)

@register(SDFString, Chem.Conformer)
def str_to_rdkit(sdf:SDFString) -> Chem.Conformer:
    with tf.NamedTemporaryFile(delete=False) as file:
        file.write(sdf.string)

    with Chem.SDMolSupplier(file.name) as suppl:
        for mol in suppl:
            conf = mol.GetConformer(0)
            try:
                os.remove(file.name)
            except OSError:
                ...
            return conf

@register(GenericMol, ASEMol)
def aps_to_ase(mol:GenericMol):
    return ASEMol(
        ase.Atoms(
            mol.symbols,
            mol.positions,
            charges=mol.charges
        ),
        bonds=mol.bonds
    )

@register(ASEMol, GenericMol)
def ase_to_aps(mol:ASEMol):
    return GenericMol(
        mol.mol.symbols,
        mol.mol.positions,
        mol.mol.get_charges(),
        bonds=mol.bonds
    )