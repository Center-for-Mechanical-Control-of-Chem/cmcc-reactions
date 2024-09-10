
import ase, numpy as np

__all__ = [
    "GenericMol",
    "MassWeightedMol",
    "SDFString",
    "ASEMol",
    "ASEMassWeightedMol"
]

class GenericMol:
    """
    An interchange type between types
    """
    def __init__(self, symbols, positions, charges, bonds=None):
        self.symbols = symbols
        self.positions = positions
        self.charges = charges
        self.bonds = [] if bonds is None else tuple(bonds)
    def copy(self):
        return type(self)(
            self.symbols,
            np.asanyarray(self.positions).copy(),
            self.charges,
            bonds=self.bonds
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
    def __init__(self, ase_mol:ase.Atoms, charges, bonds=None):
        self.mol = ase_mol
        self.charges = charges
        self.bonds = [] if bonds is None else tuple(bonds)

    def copy(self):
        return type(self)(self.mol.copy(), np.asanyarray(self.charges).copy(), bonds=self.bonds)

class ASEMassWeightedMol(ASEMol):
    """
    A wrapper for ASE to try to track bonds and charge w/ mass weighting
    """