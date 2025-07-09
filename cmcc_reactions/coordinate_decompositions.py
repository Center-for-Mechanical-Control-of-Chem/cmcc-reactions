import numpy as np
from McUtils.Data import UnitsData
import McUtils.Numputils as nput
import McUtils.Devutils as dev
import McUtils.Coordinerds as coordops
from Psience.Molecools import Molecule
from Psience.Modes import NormalModes

__all__ = [
    "get_force_projections"
]

def project_conformer(conformer_geom, internals, forces):
    if dev.is_dict_like(internals):
        keys = list(internals.keys())
        internals = internals.values()
    else:
        keys = None
    tf = nput.internal_coordinate_tensors(
        conformer_geom,
        internals
    )
    return keys, forces @ tf

def project_modes(modes, internals):
    modes = modes.remove_mass_weighting()
    return project_conformer(modes.origin, internals, modes.coords_by_modes)


def construct_force_modes(conf:Molecule, forces):
    mw_forces = forces @ conf.get_gmatrix(power=1 / 2)
    mw_forces = mw_forces / np.linalg.norm(mw_forces, axis=1)[:, np.newaxis]
    return NormalModes(
        conf.coords.system,
        mw_forces.T,
        inverse=mw_forces,
        freqs=np.ones(len(forces)),
        origin=conf.coords,
        masses=conf.masses,
        mass_weighted=True
    )

def get_force_projections(conf:Molecule, forces, internals=None, get_labels=True):
    if not isinstance(forces, NormalModes):
        forces = construct_force_modes(conf, forces)
    if internals is None:
        internals = sum(
            coordops.get_stretch_coordinate_system([b[:2] for b in conf.bonds]),
            []
        )

        if get_labels:
            labels = conf.edge_graph.get_label_types()
            internals = {
                c: coordops.get_coordinate_label(
                    c,
                    labels
                )
                for c in internals
            }

    return project_modes(forces, internals)




