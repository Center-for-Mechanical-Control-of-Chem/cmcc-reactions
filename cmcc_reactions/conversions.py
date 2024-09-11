
import functools, numpy as np, tempfile as tf, os, io
import collections, weakref

import ase
from rdkit import Chem
from McUtils.Data import AtomData
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
        if not isinstance(from_type, str):
            from_type = from_type.__qualname__
        if not isinstance(to_type, str):
            to_type = to_type.__qualname__
        if from_type not in converters: converters[from_type] = {}
        converters[from_type][to_type] = conversion

def bfs_search(converter_graph, from_type, target_type):
    bfs_queue = collections.deque()
    bfs_queue.append([from_type])
    visited = set()

    while bfs_queue:
        path = bfs_queue.popleft()
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


_conversion_cache = weakref.WeakKeyDictionary() # convert cost should only be paid once
def convert(mol, target_type:type) -> 'target_type':
    to_type = target_type.__qualname__
    # if mol in _conversion_cache and to_type in _conversion_cache[mol]:
    #     return _conversion_cache[mol][to_type]

    from_type = type(mol).__qualname__
    if from_type == to_type:
        return mol

    converter = find_conversion(from_type, to_type)

    # if mol not in _conversion_cache:
    #     _conversion_cache[mol] = {}
    # _conversion_cache[mol][to_type] = mol
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

    atom_map = [
        atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
    ]

    return GenericMol(
        symbols=sym,
        positions=pos,
        charges=charges,
        bonds=bonds,
        atom_map=atom_map
    )

def _format_counts_line(npos, nbond):
    return "{:3.0f}{:3.0f}{:3.0f}     0  0  0  0  0  0999 V2000".format(npos, nbond, 0)
@register(GenericMol, SDFString)
def mol_to_str(mol:GenericMol):
    block = []
    block.append("MOL_NAME") # tbd
    block.append("     RDKit          3D") # tbd
    block.append("")

    npos = len(mol.positions)
    nbond = len(mol.bonds)

    block.append(_format_counts_line(npos, nbond))
    for (x,y,z), sym in zip(
        mol.positions,
        mol.symbols
    ):
        block.append(
            "{:10.5f}{:10.5f}{:10.5f} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0".format(
                x, y, z, sym
            )
        )
    for i,j,t in mol.bonds:
        block.append("{:3}{:3}{:3}  0  0  0  0".format(i+1,j+1,t))
    for i,charge in enumerate(mol.charges):
        if charge != 0:
            block.append("M CHG 1 {} {}".format(i+1, charge))
    block.append("M  END")
    block.append("> <CMCC_ATOM_MAP>")
    block.append(" ".join(str(x) for x in mol.atom_map))
    block.append("")
    return SDFString("\n".join(block))

@register(SDFString, Chem.Conformer)
def str_to_rdkit(sdf:SDFString) -> Chem.Conformer:
    # with tf.NamedTemporaryFile("w+", delete=False) as file:
    #     file.write(sdf.string)
    sdf = io.BytesIO(sdf.string.encode())
    mol = next(Chem.ForwardSDMolSupplier(sdf, sanitize=False, removeHs=False))
    atom_map = mol.GetProp('CMCC_ATOM_MAP')
    atom_map = [int(i) for i in atom_map.split(" ")]
    for atom,i in zip(mol.GetAtoms(), atom_map):
        atom.SetAtomMapNum(i)

    conf = mol.GetConformer(0)

    return conf

@register(GenericMol, ASEMol)
def aps_to_ase(mol:GenericMol):
    return ASEMol(
        ase.Atoms(
            mol.symbols,
            mol.positions
        ),
        mol.charges,
        bonds=mol.bonds,
        atom_map=mol.atom_map
    )

@register(ASEMol, GenericMol)
def ase_to_aps(mol:ASEMol):
    return GenericMol(
        mol.mol.symbols,
        mol.mol.positions,
        mol.charges,
        bonds=mol.bonds,
        atom_map=mol.atom_map
    )

@register(GenericMol, MassWeightedMol)
def aps_to_mw(mol:GenericMol):
    masses = [AtomData[s]["Mass"] for s in mol.symbols]
    struct = np.asarray(mol.positions)
    mw = struct * np.sqrt(masses)[:, np.newaxis]
    return MassWeightedMol(
        mol.symbols,
        mw,
        mol.charges,
        bonds=mol.bonds,
        atom_map=mol.atom_map
    )

@register(MassWeightedMol, GenericMol)
def aps_to_mw(mol:MassWeightedMol):
    masses = [AtomData[s]["Mass"] for s in mol.symbols]
    struct = np.asarray(mol.positions)
    mw = struct / np.sqrt(masses)[:, np.newaxis]
    return GenericMol(
        mol.symbols,
        mw,
        mol.charges,
        bonds=mol.bonds,
        atom_map=mol.atom_map
    )

@register(MassWeightedMol, ASEMassWeightedMol)
def aps_to_ase(mol:MassWeightedMol):
    return ASEMassWeightedMol(
        ase.Atoms(
            mol.symbols,
            mol.positions
        ),
        charges=mol.charges,
        bonds=mol.bonds,
        atom_map=mol.atom_map
    )

@register(ASEMassWeightedMol, MassWeightedMol)
def ase_to_aps(mol:ASEMassWeightedMol):
    return MassWeightedMol(
        mol.mol.symbols,
        mol.mol.positions,
        mol.charges,
        bonds=mol.bonds,
        atom_map=mol.atom_map
    )
