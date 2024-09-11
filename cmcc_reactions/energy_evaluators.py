
import abc, io

import ase, torch, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.optimize import BFGS
from McUtils.Data import AtomData

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

    def get_calculator(self):
        raise NotImplementedError("`get_calculator` not defined for {}".format(
            type(self).__name__
        ))

class MassWeightedCalculator:
    def __init__(self, core_calc):
        self.calc = core_calc
        self.results = None

    all_change = [ # from aimnet2ase
        'positions',
        'numbers',
        'cell',
        'pbc',
        'initial_charges',
        'initial_magmoms',
    ]
    default_properties = ['energy']
    def calculate(self, atoms:ase.Atoms=None, properties=None, system_changes=None):
        if system_changes is None:
            system_changes = self.all_change
        if properties is None:
            properties = self.default_properties

        masses = [AtomData[s]["Mass"] for s in atoms.symbols]
        atoms = atoms.copy()
        sqm = np.sqrt(masses)[:, np.newaxis]
        atoms.set_positions(atoms.get_positions() / sqm)

        self.calc.calculate(atoms, properties=properties, system_changes=system_changes)
        res = self.calc.results
        if 'forces' in res:
            res['forces'] = res['forces'] / sqm
        self.results = res

    def copy(self):
        return type(self)(type(self.calc)(self.calc.base_calc))

    def set_charge(self, charge):
        return self.calc.set_charge(charge)

class AIMNetEnergyEvaluator(EnergyEvaluator):
    def __init__(self, model='aimnet2', mass_weighted=False):
        self.model = model
        self.calc = self.load_class()(self.model)
        self.mass_weighted = mass_weighted
        if mass_weighted:
            self.calc = MassWeightedCalculator(self.calc)

    def load_class(self):
        try:
            from aimnet2ase import AIMNet2Calculator as calc
        except ImportError:
            from aimnet2calc import AIMNet2ASE as calc
        return calc

    def get_calculator(self):
        if isinstance(self.calc, MassWeightedCalculator):
            return self.calc.copy()
        else:
            return self.load_class()(self.calc.base_calc)
    def calculate_energy(self, mol):
        if self.mass_weighted:
            mol = convert(mol, ASEMassWeightedMol)
        else:
            mol = convert(mol, ASEMol) # type: ASEMol
        mol.mol.calc = self.calc
        self.calc.set_charge(np.sum(mol.charges))
        self.calc.calculate(mol.mol, ['energy'])
        energy = self.calc.results['energy']
        mol.mol.calc = None
        return energy

    convergence_criterion = 0.001
    def optimize_structure(self, mol):
        mol = convert(mol, ASEMol) # type: ASEMol
        bonds = mol.bonds
        ase_mol = mol.mol
        self.calc.set_charge(np.sum(mol.charges))
        ase_mol.calc = self.calc

        opt_rea = BFGS(ase_mol, logfile=io.StringIO())
        opt_rea.run(fmax=self.convergence_criterion)
        return ASEMol(
            ase_mol,
            charges=mol.charges,
            bonds=bonds,
            atom_map=mol.atom_map
        )

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

