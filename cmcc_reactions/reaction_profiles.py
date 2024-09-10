

import abc, dataclasses, io, numpy as np

from ase.mep import DyNEB
from ase.optimize import BFGS

from .energy_evaluators import *
from .components import *
from .conversions import *
from .mol_types import *

__all__ = [
    "ReactionProfileGenerator",
    "DyNEBProfileGenerator"
]

@dataclasses.dataclass
class ReactionProfileData:
    initial_profile: 'list[ASEMol]'
    optimized_profile: 'list[ASEMol]'

class ReactionProfileGenerator(metaclass=abc.ABCMeta):
    def __init__(self,
                 reactants:ReactionComponentSet, products:ReactionComponentSet,
                 energy_evaluator:EnergyEvaluator,
                 preoptimize=True,
                 mass_weight=True
                 ):

        self.preoptimize = preoptimize
        reactants = reactants.unified_component
        products = products.unified_component
        if preoptimize:
            reactants = energy_evaluator.optimize_structure(reactants)
            products = energy_evaluator.optimize_structure(products)
        if mass_weight:
            reactants = convert(reactants, MassWeightedMol)
            products = convert(products, MassWeightedMol)
        self.reactants = reactants
        self.products = products

        self.evaluator = energy_evaluator
        self.mass_weighted = mass_weight

    @abc.abstractmethod
    def generate_profile(self) -> 'ReactionProfileData':
        ...

    @classmethod
    def from_smiles(cls, rxn_smiles, energy_evaluator, num_conformers=1, **opts):
        reactant_smiles, product_smiles = rxn_smiles.split(">>")
        reactants = ReactionComponentSet.from_smiles(reactant_smiles,
                                                     energy_evaluator=energy_evaluator,
                                                     num_conformers=num_conformers
                                                     )
        products = ReactionComponentSet.from_smiles(product_smiles,
                                                    energy_evaluator=energy_evaluator,
                                                    num_conformers=num_conformers
                                                    )
        return cls(reactants, products, energy_evaluator, **opts)

class DyNEBProfileGenerator(ReactionProfileGenerator):

    def __init__(self,
                 reactants:ReactionComponentSet, products:ReactionComponentSet,
                 energy_evaluator:EnergyEvaluator,
                 preoptimize=True,
                 mass_weight=None,
                 images=5,
                 climb=False,
                 dynamic_relaxation=False
                 ):
        if mass_weight is None:
            mass_weight = energy_evaluator.mass_weighted #TODO: add mass_weighted to other evaluators
        super().__init__(
            reactants, products, energy_evaluator,
            preoptimize=preoptimize,
            mass_weight=mass_weight
        )

        if not isinstance(images, (int, np.integer)):
            self.neb = DyNEB(
                images,
                climb=climb,
                fmax=AIMNetEnergyEvaluator.convergence_criterion,
                dynamic_relaxation=dynamic_relaxation
            )
        else:
            if mass_weight:
                ase_type = ASEMassWeightedMol
            else:
                ase_type = ASEMol
            reactants = convert(self.reactants, ase_type)
            products = convert(self.products, ase_type)
            nimg = images
            images = (
                    [reactants]
                    + [reactants.copy() for _ in range(nimg)]
                    + [products]
            )
            self.neb = DyNEB(
                [img.mol for img in images],
                climb=climb,
                fmax=AIMNetEnergyEvaluator.convergence_criterion,
                dynamic_relaxation=dynamic_relaxation
            )
            self.neb.interpolate()
            for img in self.neb.images:
                img.calc = energy_evaluator.get_calculator()

    def generate_profile(self) -> 'ReactionProfileData':
        initial_profile = [
            ASEMol(mol.copy(), self.reactants.charges)
            for mol in self.neb.images
        ]

        optimizer = BFGS(self.neb, logfile=io.StringIO())
        optimizer.run(fmax=AIMNetEnergyEvaluator.convergence_criterion, steps=1000)

        final_profile = [
            ASEMol(mol, self.reactants.charges)
            for mol in self.neb.images
        ]
        return ReactionProfileData(
            initial_profile,
            final_profile
        )





