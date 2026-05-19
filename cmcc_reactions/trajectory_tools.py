
import collections
import os.path

import numpy as np

from . import reaction_data_schema as schema
from . import utils
from . import generate_reaction_products as gen_prods

from McUtils.Data import UnitsData, BondData
import McUtils.Numputils as nput
import McUtils.Plots as plt
from Psience.Molecools import Molecule
import Psience.Plots as psiplot

__all__ = [
    "DielsAlderReactionTrajectory"
]

def get_critical_points(trajectory, energies=None, initial=None):
    if energies is None:
        energies = [g.calculate_energy() for g in trajectory]

    product_idx = np.argmin(energies)
    ts_idx = np.argmax(energies)
    if product_idx > ts_idx:
        react_idx = np.argmin(energies[:ts_idx])
    else:
        react_idx = np.argmin(energies[ts_idx:]) + ts_idx
    return energies, (ts_idx, react_idx, product_idx)

def centroid_distance(traj, bonds):
    traj = np.asanyarray(traj)
    a1, a2 = np.array(bonds).T
    return nput.pts_norms(
        np.average(traj[:, a1], axis=-2),
        np.average(traj[:, a2], axis=-2)
    )

def bond_average_distance(traj, bonds):
    traj = np.asanyarray(traj)
    a1, a2 = np.array(bonds).T
    return np.average(nput.pts_norms(traj[:, a1], traj[:, a2]), axis=-1)

default_cc_single_distance = 1.54#BondData["C", "C"]
def cc_single_normalized_distance(traj, bonds):
    return default_cc_single_distance - bond_average_distance(traj, bonds)

def bond_centroid_deviation_distance(traj, bonds):
    return centroid_distance(traj, bonds) - bond_average_distance(traj, bonds)

def bond_1(traj, bonds):
    traj = np.asanyarray(traj)
    return nput.pts_norms(traj[:, bonds[0][0]], traj[:, bonds[0][1]])

def bond_2(traj, bonds):
    traj = np.asanyarray(traj)
    return nput.pts_norms(traj[:, bonds[1][0]], traj[:, bonds[1][1]])

def dienophile_distance(traj, bonds):
    traj = np.asanyarray(traj)
    _, (i, j) = np.array(bonds).T
    return nput.pts_norms(traj[:, i], traj[:, j])

def incremental_rmsds(traj, bonds=None, sel=None):
    traj = np.asanyarray(traj)
    if sel is None and bonds is not None:
        sel = np.asanyarray(bonds).flatten()
    if sel is not None:
        traj = traj[..., sel, :]
    disps = np.diff(traj, axis=0).reshape((len(traj)-1, -1))
    rmsds = np.linalg.norm(disps, axis=-1)
    return np.cumsum(np.concatenate([[0], rmsds]), axis=0)

metric_label_map = {
    centroid_distance:'Centroid Distance',
    bond_average_distance:r'$r_\text{avg.}$',
    cc_single_normalized_distance:r'$\Delta r_\text{avg.}$',
    bond_1:r'$r_{1,3}$',
    bond_2:r'$r_{2,4}$',
    dienophile_distance:r'$r_{\text{C=C}}$'
}
default_distance_metric = cc_single_normalized_distance
def resolve_distance_metric(distance_metric):
    if distance_metric is None:
        distance_metric = default_distance_metric
    return distance_metric
def plot_reaction_profile(
        coordinates,
        energies,
        distance_metric=None,
        metric_label=None,
        bonds=((0, 2), (1, 3)),
        return_metrics=False,
        **opts):
    energies, (ts, r, p) = get_critical_points(None, energies)
    energies = np.asanyarray(energies)
    if distance_metric is None:
        distance_metric = cc_single_normalized_distance
    if metric_label is None:
        metric_label = metric_label_map.get(distance_metric)
        if metric_label is None:
            metric_label = distance_metric.__name__
    x1 = distance_metric(coordinates, bonds) * UnitsData.convert("BohrRadius", "Angstroms")
    figure = plt.Plot(
        x1,
        (energies - energies[r]) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
        **(dict(
            axes_labels=[
                metric_label + r" ($\AA$)",
                "E (kcal mol$^{-1}$)"
            ]
        ) | opts)
    )
    if return_metrics:
        return figure, x1
    else:
        return figure

def plot_metric_profile(
        coordinates,
        distance_metric_1,
        distance_metric_2,
        metric_label_1=None,
        metric_label_2=None,
        bonds=((0, 2), (1, 3)),
        **opts):
    if metric_label_1 is None:
        metric_label_1 = metric_label_map.get(distance_metric_1)
        if metric_label_1 is None:
            metric_label_1 = metric_label_1.__name__
    if metric_label_2 is None:
        metric_label_2 = metric_label_map.get(distance_metric_2)
        if metric_label_2 is None:
            metric_label_2 = metric_label_2.__name__
    x1 = distance_metric_1(coordinates, bonds) * UnitsData.convert("BohrRadius", "Angstroms")
    x2 = distance_metric_2(coordinates, bonds) * UnitsData.convert("BohrRadius", "Angstroms")
    return plt.Plot(
        x1,
        x2,
        **(dict(
            axes_labels=[
                metric_label_1 + r" ($\AA$)",
                metric_label_2 + r" ($\AA$)",
            ]
        ) | opts)
    )

def plot_reaction_lines(
        coordinates,
        energies,
        connect=True,
        baseline=None,
        **opts
):
    if baseline is None:
        baseline = energies[0]
    energies = np.asanyarray(energies) - baseline
    energies = energies * UnitsData.convert("Hartrees", "Kilocalories/Mole")
    if coordinates[0] < coordinates[-1]:
        coordinates = np.flip(coordinates)
        energies = np.flip(energies)
    # {
    #     'r': {
    #         'x': 0,
    #         'y': [0., 1.85532628]
    #     },
    #     'ts':
    #         {
    #             'x': .5,
    #             'y': [19.7746029, 19.80986744]
    #         },
    #     'prod': {
    #         'x': 5,
    #         'y': [-25.01321145379916]
    #     }
    # }
    return psiplot.plot_energy_levels(
        [
            {'x': c, 'y': [e]}
            for c, e in zip(coordinates, energies)
        ],
        connect=connect,
        **opts
    )

class DielsAlderReactionTrajectory:
    #TODO: split this into a ReactionTrajectory base class
    #      and subclass in specifically the `diene_atoms` etc.
    def __init__(self, atoms, structures,
                 energies=None,
                 ts_index=None,
                 reactant_index=None,
                 product_index=None,
                 energy_evaluator=None,
                 distance_units=None,
                 diene_atoms=(2, 3, 4, 5),
                 dienophile_atoms=(0, 1)
                 ):
        self.atoms = atoms
        self.structures = np.asanyarray(structures)
        self._energies = energies
        self._mols = [None] * len(self.structures)
        self._ts_idx = ts_index
        self._reactant_idx = reactant_index
        self._product_idx = product_index
        self.energy_evaluator = energy_evaluator
        self.distance_units = distance_units
        self.diene_atoms = diene_atoms
        self.dienophile_atoms = dienophile_atoms

    def load_mol(self, i):
        if self._mols[i] is None:
            struct = self.structures[i]
            if self.distance_units is not None:
                struct = struct * UnitsData.convert(self.distance_units, "BohrRadius")
            self._mols[i] = Molecule(
                self.atoms,
                struct,
                energy_evaluator=self.energy_evaluator
            )
        return self._mols[i]

    @property
    def energies(self):
        if self._energies is None:
            self._energies = [g.calculate_energy() for g in self.mols]
        return self._energies

    @property
    def mols(self):
        for i in range(len(self.structures)):
            self.load_mol(i)
        return self._mols

    @property
    def ts_index(self):
        if self._ts_idx is None:
            _, (self._ts_idx, self._reactant_idx, self._product_idx) = get_critical_points(None, self.energies)
        return self._ts_idx
    @property
    def reactant_index(self):
        if self._reactant_idx is None:
            _, (self._ts_idx, self._reactant_idx, self._product_idx) = get_critical_points(None, self.energies)
        return self._reactant_idx
    @property
    def product_index(self):
        if self._product_idx is None:
            _, (self._ts_idx, self._reactant_idx, self._product_idx) = get_critical_points(None, self.energies)
        return self._product_idx
    @property
    def transition_state(self):
        return self.load_mol(self.ts_index)
    @property
    def reactant(self):
        return self.load_mol(self.reactant_index)
    @property
    def product(self):
        return self.load_mol(self.product_index)

    @classmethod
    def from_trajectory_data(cls,
                             trajectory_data,
                             structures=None,
                             energies=None,
                             energy_evaluator=None,
                             which='final',
                             **etc):
        if isinstance(trajectory_data, str):
            trajectory_data = utils.read_namedtuple(trajectory_data, gen_prods.ReoptimizedTrajectoryData)
        elif isinstance(trajectory_data, dict):
            if 'trajectory_energies' in trajectory_data:
                trajectory_data = gen_prods.TrajectoryData(
                    atoms=trajectory_data['atoms'],
                    coordinates=trajectory_data['trajectory'],
                    energies=trajectory_data['trajectory_energies'],
                    rmsds=None,
                    evaluator=energy_evaluator
                )
            elif 'initial_trajectory_energies' in trajectory_data:
                trajectory_data = gen_prods.ReoptimizedTrajectoryData(
                    atoms=trajectory_data['atoms'],
                    initial_trajectory=trajectory_data['initial_trajectory'],
                    initial_energies=trajectory_data['initial_trajectory_energies'],
                    final_trajectory=trajectory_data['refined_trajectory'],
                    final_energies=trajectory_data['refined_trajectory_energies'],
                    initial_rmsds=None,
                    final_rmsds=None,
                    raw_pre_energies=None,
                    raw_pre_sampling=None,
                    evaluator=energy_evaluator
                )
            elif 'final_energies' in trajectory_data:
                trajectory_data = gen_prods.ReoptimizedTrajectoryData(**trajectory_data)
            else:
                trajectory_data = gen_prods.TrajectoryData(**trajectory_data)

        if energies is None:
            if hasattr(trajectory_data, 'final_energies'):
                if which == 'final':
                    energies = trajectory_data.final_energies
                else:
                    energies = trajectory_data.initial_energies
            else:
                energies = trajectory_data.energies

        if structures is None:
            if hasattr(trajectory_data, 'final_energies'):
                if which == 'final':
                    structures = trajectory_data.final_trajectory
                else:
                    structures = trajectory_data.initial_trajectory
            else:
                structures = trajectory_data.coordinates

        if energy_evaluator is None:
            energy_evaluator = trajectory_data.evaluator
        return cls(
            trajectory_data.atoms,
            structures=structures,
            energies=energies,
            energy_evaluator=energy_evaluator,
            **etc
        )

    @classmethod
    def from_file(cls, file, **etc):
        return cls.from_trajectory_data(
            utils.read_namedtuple(file),
            **etc
        )

    def save(self, output_dir, info_file='profile.json', **etc):
        if os.path.splitext(output_dir)[-1].startswith('.'):
            output_dir, info_file = os.path.split(output_dir)
        data = gen_prods.write_trajectory(
            output_dir,
            self.mols,
            **(
                    dict(
                        energies=self.energies,
                        info_file=info_file) | etc
            )
        )
        return os.path.join(output_dir, info_file)

    def plot_profile(self,
                     distance_metric=None,
                     metric_label=None,
                     bonds=((0, 2), (1, 3)),
                     return_metrics=False,
                     **opts
                     ):
        return plot_reaction_profile(
            self.structures,
            self.energies,
            distance_metric=distance_metric,
            metric_label=metric_label,
            bonds=bonds,
            return_metrics=return_metrics,
            **opts
        )

    def compare_profiles(self, other,
                         distance_metric=None,
                         metric_label=None,
                         bonds=((0, 2), (1, 3)),
                         figure=None,
                         comparison_styles=None,
                         labels=None,
                         return_metrics=False,
                         **opts):
        figure, metrics = self.plot_profile(
            distance_metric=distance_metric,
            metric_label=metric_label,
            bonds=bonds,
            figure=figure,
            return_metrics=True,
            **(opts | dict(label=labels[0] if labels is not None else None))
        )
        if comparison_styles is None:
            comparison_styles = {'linestyle':'dashed'}
        other.plot_profile(
            distance_metric=distance_metric,
            metric_label=metric_label,
            bonds=bonds,
            figure=figure,
            **(opts | comparison_styles | dict(label=labels[1] if labels is not None else None))
        )
        if return_metrics:
            return figure, metrics
        else:
            return figure

    def animate_trajectory(self,
                           bonds=None,
                           **opts):
        anim = None
        base_mol = self.load_mol(0)
        if bonds is None:
            try:
                anim = base_mol.plot(
                    self.structures,
                    bonds='recompute',
                    **opts
                )
            except ValueError:
                anim = None
        if anim is None:
            anim = base_mol.plot(
                self.structures,
                bonds=bonds,
                **opts
            )
        return anim

def compare_profiles(
        trajectory_data,
        **opts
):
    if isinstance(trajectory_data, str):
        d1 = DielsAlderReactionTrajectory.from_file(
            trajectory_data,
            which='final',
        )
        d2 = DielsAlderReactionTrajectory.from_file(
            trajectory_data,
            which='initial'
        )
    else:
        d1 = DielsAlderReactionTrajectory.from_trajectory_data(
            trajectory_data,
            which='final',
        )
        d2 = DielsAlderReactionTrajectory.from_trajectory_data(
            trajectory_data,
            which='initial'
        )
    return d1.compare_profiles(d2, **opts)

# def compare_profiles_from_file(
#         trajectory_file,
#         **opts
# ):
#     return DielsAlderReactionTrajectory.from_file(
#         trajectory_file,
#         which='final',
#     ).compare_profiles(
#         DielsAlderReactionTrajectory.from_file(
#             trajectory_file,
#             which='initial'
#         ),
#         **opts
#     )

# def plot_comp_traj(traj_data,
#                    comp_data=None,
#                    use_rmsd=None,
#                    distance_metric=None,
#                    metric_label=None,
#                    **opts):
#     min_e = np.min(np.concatenate([traj_data.final_energies[:5], traj_data.initial_energies[:5]]))
#     if distance_metric is None:
#         distance_metric = dienophile_distance
#     elif use_rmsd is None:
#         use_rmsd = False
#     if use_rmsd:
#         if metric_label is None:
#             metric_label = "RMSD"
#         x1 = np.array(traj_data.final_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
#     else:
#         if metric_label is None:
#             metric_label = metric_label_map.get(distance_metric)
#             if metric_label is None:
#                 metric_label = distance_metric.__name__
#         x1 = distance_metric(traj_data.final_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
#                                                                                                "Angstroms")
#     f1 = plt.Plot(
#         x1,
#         (traj_data.final_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
#         **(dict(
#             axes_labels=[
#                 metric_label + r" ($\AA$)",
#                 "E (kcal mol$^{-1}$)"
#             ]
#         ) | opts)
#     )
#     if comp_data is None:
#         if use_rmsd:
#             x2 = np.array(traj_data.initial_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
#         else:
#             x2 = distance_metric(traj_data.initial_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
#                                                                                                      "Angstroms")
#         plt.Plot(
#             x2,
#             (traj_data.initial_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
#             figure=f1,
#             linestyle='dashed'
#         )
#     else:
#         if use_rmsd:
#             x2 = np.array(comp_data.final_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
#         else:
#             x2 = distance_metric(comp_data.final_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
#                                                                                                    "Angstroms")
#         plt.Plot(
#             x2,
#             (comp_data.final_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
#             figure=f1,
#             linestyle='dashed'
#         )
#     return f1
