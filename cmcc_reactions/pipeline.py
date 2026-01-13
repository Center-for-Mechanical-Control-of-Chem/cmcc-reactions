from __future__ import annotations

from Psience.Molecools import Molecule
import numpy as np

def apply_distortion_library_distortions(
        atoms:list[str],
        reactant_coords:np.ndarray,
        ts_coordss:np.ndarray,
        distortion_specs=None,
        energy_evaluator=None,
        **other_options
):
    ...

def apply_explicit_force_transition_state_search(
        atoms,
        reactant_coords: np.ndarray,
        ts_coordss: np.ndarray,
):
    ...

class ForceAdaptedReaction:
    def __init__(self,
                 reactant:Molecule|str,
                 transition_state:Molecule|str):
        self.reactant = Molecule.construct(reactant)
        self.transition_state = Molecule.construct(transition_state)
        self.energy_evaluator = ...