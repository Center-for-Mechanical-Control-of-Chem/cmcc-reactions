
import ase

__all__ = [
    "GenericMol",
    "SDFString",
    "ASEMol",
]

class GenericMol:
    """
    An interchange type between types
    """
    def __init__(self, symbols, positions, charges, bonds=None):
        self.symbols = symbols
        self.positions = positions
        self.charges = charges
        self.bonds = [] if bonds is None else bonds

class SDFString:
    def __init__(self, string):
        self.string = string

class ASEMol:
    """
    A wrapper for ASE to try to track bonds and charge
    """
    def __init__(self, ase_mol:ase.Atoms, bonds=None):
        self.mol = ase_mol
        self.bonds = [] if bonds is None else bonds