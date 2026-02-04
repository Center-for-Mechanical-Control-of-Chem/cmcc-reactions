from __future__ import annotations

from Psience.Molecools import Molecule
import numpy as np

def apply_distortion_library_distortions(
        atoms:list[str],
        reactant_coords:np.ndarray,
        ts_coords:np.ndarray,
        distortion_specs=None,
        energy_evaluator='pyscf', 
        path='gv_job', 
        theory: dict|None =None, 
        rxn_type='DielsAlder',
        **other_options
):
    
    from goodvibs.rxn import Reaction
    
    Reaction.import_xyzs(xyzs=[reactant_coords, ts_coords], ats=atoms, path=path, theory=theory)

    rxn = Reaction.from_path(path, type=rxn_type, software=energy_evaluator) # Default is pySCF using lowest L.o.T.

    if distortion_specs:
        for coordinate in distortion_specs:
            rxn.add_distortion(coordinate)

    rxn.scan_by_coordinate()
    rxn.read_coordinate_scan()

    rxn.scan_by_force()
    rxn.read_force_scan()

    return rxn.export()

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