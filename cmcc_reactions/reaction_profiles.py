

import abc, dataclasses, io, numpy as np

from ase.mep import DyNEB
from ase.optimize import BFGS

from .energy_evaluators import *
from .components import *
from .conversions import *
from .mol_types import *
from .atom_maps import *

__all__ = [
    "ReactionProfileGenerator",
    "DyNEBProfileGenerator"
]

@dataclasses.dataclass
class ReactionProfileData:
    initial_profile: 'list[ASEMol]'
    optimized_profile: 'list[ASEMol]'
    energy_evaluator: 'EnergyEvaluator'

class ReactionProfileGenerator(metaclass=abc.ABCMeta):
    def __init__(self,
                 reactants,
                 products,
                 energy_evaluator:EnergyEvaluator,
                 preoptimize='products',
                 mass_weight=True
                 ):
        self.preoptimize = preoptimize
        if preoptimize:
            if preoptimize is True or preoptimize == 'reactants':
                reactants = energy_evaluator.optimize_structure(reactants)
            if preoptimize is True or preoptimize == 'products':
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
    def parse_smiles(cls,
                     rxn_smiles, energy_evaluator,
                     num_conformers=1,
                     atom_mapper='chython',
                     add_implicit_hydrogens=None,
                     preoptimize='products'
                     ):
        if add_implicit_hydrogens is None:
            add_implicit_hydrogens = "H" not in rxn_smiles
        if atom_mapper is not None:
            if isinstance(atom_mapper, str):
                if atom_mapper == "chython":
                    atom_mapper = ChythonAtomMapper(add_implicit_hydrogens=add_implicit_hydrogens)
                else:
                    raise ValueError(f"unknown atom mapper '{atom_mapper}'")
            rxn_smiles = atom_mapper.apply(rxn_smiles)
        reactant_smiles, product_smiles = rxn_smiles.split(">>")
        products = ReactionComponentSet.from_smiles(product_smiles,
                                                    energy_evaluator=energy_evaluator,
                                                    num_conformers=num_conformers,
                                                    add_implicit_hydrogens=add_implicit_hydrogens,
                                                    optimize=False
                                                    )
        if preoptimize is True or preoptimize == 'products':
            products.unified_component = energy_evaluator.optimize_structure(products.unified_component)
        reactants = ReactionComponentSet.from_smiles(reactant_smiles,
                                                     energy_evaluator=energy_evaluator,
                                                     num_conformers=num_conformers,
                                                     add_implicit_hydrogens=add_implicit_hydrogens,
                                                     reference_structure=products.unified_component
                                                     )
        if preoptimize is True or preoptimize == 'reactants':
            reactants.unified_component = energy_evaluator.optimize_structure(reactants.unified_component)

        return reactants, products

    @classmethod
    def from_smiles(cls,
                    rxn_smiles, energy_evaluator,
                    num_conformers=1,
                    atom_mapper='chython',
                    add_implicit_hydrogens=None,
                    **opts
                    ):
        reactants, products = cls.parse_smiles(
            rxn_smiles, energy_evaluator,
            num_conformers=num_conformers,
            atom_mapper=atom_mapper,
            add_implicit_hydrogens=add_implicit_hydrogens
        )
        return cls(reactants.unified_component, products.unified_component, energy_evaluator, **opts)

class DyNEBProfileGenerator(ReactionProfileGenerator):

    def __init__(self,
                 reactants:ReactionComponentSet, products:ReactionComponentSet,
                 energy_evaluator:EnergyEvaluator,
                 preoptimize='products',
                 mass_weight=None,
                 images=8,
                 climb=False,
                 dynamic_relaxation=False,
                 spring_constant=0.1
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
                k=spring_constant,
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
            final_profile,
            self.evaluator
        )





