
import abc
import ase, torch


__all__ = [
    "AIMNetEnergyEvaluator",
    "MMFF94EnergyEvaluator"
]
class EnergyEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate_energy(self, mol:ase.Atoms):
        ...
    def __call__(self, mol:ase.Atoms):
        return self.calculate_energy(mol)


class AIMNetEnergyEvaluator(EnergyEvaluator):
    def __init__(self, model, charge):
        ...

class MMFF94EnergyEvaluator(EnergyEvaluator):
    def __init__(self):
        ...