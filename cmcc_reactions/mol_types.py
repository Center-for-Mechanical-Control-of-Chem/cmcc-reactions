
import ase, numpy as np
from Psience.Molecools import Molecule

__all__ = [
    "GenericMol",
    "MassWeightedMol",
    "SDFString",
    "ASEMol",
    "ASEMassWeightedMol",
    "PsiMol"
]

class GenericMol:
    """
    An interchange type between types
    """
    def __init__(self, symbols, positions, charges, bonds=None, atom_map=None):

        self.symbols = symbols
        self.positions = positions
        self.charges = charges
        self.bonds = [] if bonds is None else tuple(bonds)
        self.atom_map = 1+np.arange(len(charges)) if atom_map is None else atom_map
    def copy(self):
        return type(self)(
            self.symbols,
            np.asanyarray(self.positions).copy(),
            self.charges,
            bonds=self.bonds,
            atom_map=self.atom_map
        )

class MassWeightedMol(GenericMol):
    """
    A mass-weighted form of a `GenericMol`
    """


class SDFString:
    def __init__(self, string):
        self.string = string

class ASEMol:
    """
    A wrapper for ASE to try to track bonds and charge
    """
    def __init__(self, ase_mol:ase.Atoms, charges, bonds=None, atom_map=None):
        self.mol = ase_mol
        self.charges = charges
        self.bonds = [] if bonds is None else tuple(bonds)
        self.atom_map = 1+np.arange(len(charges)) if atom_map is None else atom_map

    def copy(self):
        return type(self)(self.mol.copy(), np.asanyarray(self.charges).copy(), bonds=self.bonds, atom_map=self.atom_map)

class ASEMassWeightedMol(ASEMol):
    """
    A wrapper for ASE to try to track bonds and charge w/ mass weighting
    """

class PsiMol:
    """
    A wrapper for ASE to try to track bonds and charge
    """
    def __init__(self, psi_mol:Molecule, charges, atom_map=None):
        self.mol = psi_mol
        self.charges = charges
        self.atom_map = 1+np.arange(len(charges)) if atom_map is None else atom_map

    def copy(self):
        return type(self)(self.mol.copy(), np.asanyarray(self.charges).copy(), atom_map=self.atom_map)

    @property
    def atoms(self):
        return self.mol.atoms
    @property
    def coords(self):
        return self.mol.coords # convert to Angstroms?
    @property
    def bonds(self):
        return self.mol.bonds