
import abc
import ase, torch
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from aimnet2ase import AIMNet2Calculator
except ImportError:
    from aimnet2calc import AIMNet2Calculator

from ase.optimize import BFGS

from .mol_types import *
from .conversions import convert


__all__ = [
    "EnergyEvaluator",
    "AIMNetEnergyEvaluator",
    "MMFFEnergyEvaluator"
]
class EnergyEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate_energy(self, mol):
        ...
    @abc.abstractmethod
    def optimize_structure(self, mol):
        ...

class AIMNetEnergyEvaluator(EnergyEvaluator):
    def __init__(self, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model, map_location=device)
        self.calc = AIMNet2Calculator(self.model)

    def calculate_energy(self, mol):
        mol = convert(mol, ASEMol).mol # type: ASEMol
        # charge = sum(mol.charges)
        # self.calc.set_charge(charge) # TODO: apparently this is no longer necessary?
        return self.calc(mol)

    def optimize_structure(self, mol, fmax=0.001):
        opt_rea = BFGS(mol)
        opt_rea.run()
        return mol

class MMFFEnergyEvaluator(EnergyEvaluator):

    def __init__(self, mmff_variant='MMFF94s', params=None):
        self.variant = mmff_variant
        self.params = self.get_default_params() if params is None else params

    MAX_ATTEMPTS = 1000
    @classmethod
    def get_default_params(cls):
        params = AllChem.ETKDGv3()
        params.maxAttempts = cls.MAX_ATTEMPTS  # Increase the number of attempts
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True

        return params

    def calculate_energy(self, mol):
        mol = convert(mol, Chem.Conformer)# type: Chem.Conformer
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=mol.GetId())
        return ff.CalcEnergy()

    def optimize_structure(self, mol, params=None):
        mol = convert(mol, Chem.Conformer)# type: Chem.Conformer
        AllChem.MMFFOptimizeMolecule(
            mol.GetOwningMol(),
            confId=mol.GetId(),
            params=self.get_default_params() if params is None else params
        )
        return mol

