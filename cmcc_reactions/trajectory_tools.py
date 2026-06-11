from __future__ import annotations


import collections
import os.path

import numpy as np

# from . import reaction_data_schema as schema
from . import utils

from Psience.Reactions import Reaction
from McUtils.Data import UnitsData
import McUtils.Numputils as nput
import McUtils.Plots as plt
from Psience.Molecools import Molecule
import Psience.Plots as psiplot

__all__ = [
    "DielsAlderReactionTrajectory"
]

TrajectoryData = collections.namedtuple(
    "TrajectoryData",
    [
        "atoms",
        "coordinates",
        "energies",
        "gradients",
        "hessians",
        "rmsds",
        "ts_index",
        "evaluator"
    ]
)
utils.register_namedtuple(TrajectoryData,
                          defaults={
                              "ts_index": None,
                              "gradients": None,
                              "hessians": None,
                          })

def create_trajectory_data(structures, energies=None, energy_evaluator=None, rmsds=None,
                           gradients=None, hessians=None, ts_index=None
                           ):
    if energy_evaluator is None:
        energy_evaluator = structures[0].energy_evaluator
        if energies is None:
            expansions = [g.modify(energy_evaluator=energy_evaluator).calculate_energy(order=2) for g in structures]
            energies = [e[0] for e in expansions]
            if gradients is None:
                gradients = [e[1] for e in expansions]
            if hessians is None:
                hessians = [e[2] for e in expansions]
    else:
        if energies is None:
            expansions = [g.modify(energy_evaluator=energy_evaluator).calculate_energy(order=2) for g in structures]
            energies = [e[0] for e in expansions]
            if gradients is None:
                gradients = [e[1] for e in expansions]
            if hessians is None:
                hessians = [e[2] for e in expansions]

    coords = [s.coords for s in structures]
    if rmsds is None:
        rmsds = nput.incremental_eckart_rmsd(coords, masses=structures[0].masses, mass_weighted=False)

    if ts_index is None:
        _, inds = get_critical_points(structures, energies=energies, hessians=hessians)
        ts_index = inds.ts

    return TrajectoryData(
        atoms=structures[0].atoms,
        coordinates=coords,
        energies=energies,
        gradients=gradients,
        hessians=hessians,
        ts_index=ts_index,
        rmsds=rmsds,
        evaluator=energy_evaluator,
    )
def write_trajectory(output_dir, structures, energies=None, energy_evaluator=None, rmsds=None,
                     info_file='profile.json'
                     ):
    os.makedirs(output_dir, exist_ok=True)
    traj_data = create_trajectory_data(structures, energies=energies, energy_evaluator=energy_evaluator, rmsds=rmsds)
    utils.write_namedtuple(
        os.path.join(output_dir, info_file),
        traj_data
    )
    return traj_data

ReoptimizedTrajectoryData = collections.namedtuple(
    "ReoptimizedTrajectoryData",
    [
        "atoms",
        "final_trajectory",
        "final_energies",
        "final_gradients",
        "final_hessians",
        "final_ts_index",
        "final_rmsds",
        "initial_trajectory",
        "initial_energies",
        "initial_gradients",
        "initial_hessians",
        "initial_ts_index",
        "initial_rmsds",
        "raw_pre_sampling",
        "raw_pre_energies",
        "evaluator",
        "optimization_settings"
    ],
    defaults=[None]
)
utils.register_namedtuple(ReoptimizedTrajectoryData,
                          defaults={
                              "initial_gradients": None,
                              "initial_hessians": None,
                              "initial_ts_index": None,
                              "final_gradients": None,
                              "final_hessians": None,
                              "final_ts_index": None,
                          })

def create_refined_trajectory_data(initial_trajectory:TrajectoryData,
                                   final_trajectory:TrajectoryData,
                                   initial_energies=None,
                                   initial_gradients=None,
                                   initial_hessians=None,
                                   initial_ts_index=None,
                                   initial_rmsds=None,
                                   final_energies=None,
                                   final_gradients=None,
                                   final_hessians=None,
                                   final_ts_index=None,
                                   final_rmsds=None,
                                   pre_sampling=None,
                                   pre_sampling_energies=None,
                                   energy_evaluator=None,
                                   ):
    if not hasattr(initial_trajectory, 'atoms'):
        initial_trajectory = create_trajectory_data(initial_trajectory,
                                                    energies=initial_energies,
                                                    gradients=initial_gradients,
                                                    hessians=initial_hessians,
                                                    ts_index=initial_ts_index,
                                                    rmsds=initial_rmsds,
                                                    energy_evaluator=energy_evaluator)
    if not hasattr(final_trajectory, 'atoms'):
        final_trajectory = create_trajectory_data(final_trajectory,
                                                  energies=final_energies,
                                                  gradients=final_gradients,
                                                  hessians=final_hessians,
                                                  ts_index=final_ts_index,
                                                  rmsds=final_rmsds,
                                                  energy_evaluator=energy_evaluator)
    if pre_sampling is not None and not hasattr(pre_sampling, 'atoms'):
        pre_sampling = create_trajectory_data(pre_sampling,
                                              energies=pre_sampling_energies)
    return ReoptimizedTrajectoryData(
        atoms=initial_trajectory.atoms,
        final_trajectory=final_trajectory.coordinates,
        final_energies=final_trajectory.energies,
        final_gradients=final_trajectory.gradients,
        final_hessians=final_trajectory.hessians,
        final_ts_index=final_trajectory.ts_index,
        final_rmsds=final_trajectory.rmsds,
        initial_trajectory=initial_trajectory.coordinates,
        initial_energies=initial_trajectory.energies,
        initial_gradients=initial_trajectory.gradients,
        initial_hessians=initial_trajectory.hessians,
        initial_ts_index=initial_trajectory.ts_index,
        initial_rmsds=initial_trajectory.rmsds,
        raw_pre_sampling=pre_sampling.coordinates if pre_sampling is not None else None,
        raw_pre_energies=pre_sampling.energies if pre_sampling is not None else None,
        evaluator=energy_evaluator
    )
def write_refined_trajectory(output_dir,
                             initial_trajectory: TrajectoryData,
                             final_trajectory: TrajectoryData,
                             initial_energies=None,
                             initial_rmsds=None,
                             final_energies=None,
                             final_rmsds=None,
                             pre_sampling=None,
                             pre_sampling_energies=None,
                             energy_evaluator=None,
                             info_file='refined.json'):
    os.makedirs(output_dir, exist_ok=True)
    traj_data = create_refined_trajectory_data(
        initial_trajectory,
        final_trajectory,
        initial_energies=initial_energies,
        initial_rmsds=initial_rmsds,
        final_energies=final_energies,
        final_rmsds=final_rmsds,
        pre_sampling=pre_sampling,
        pre_sampling_energies=pre_sampling_energies,
        energy_evaluator=energy_evaluator
    )
    utils.write_namedtuple(
        os.path.join(output_dir, info_file),
        traj_data
    )
    return traj_data
def update_trajectory_data(traj_data:TrajectoryData|ReoptimizedTrajectoryData):
    # make it easier to update pre-scanned data
    if hasattr(traj_data, 'initial_energies'):
        traj_data:ReoptimizedTrajectoryData
        if traj_data.initial_gradients is None:
            mols = [
                Molecule(traj_data.atoms, c, energy_evaluator=traj_data.evaluator,
                         spin=1)
                for c in traj_data.initial_trajectory
            ]
            expansions = [m.calculate_energy(order=2) for m in mols]
            grads = [e[1] for e in expansions]
            hessians = [e[2] for e in expansions]
            traj_data = traj_data._replace(
                initial_gradients=grads,
                initial_hessians=hessians,
            )
        if traj_data.final_gradients is None:
            mols = [
                Molecule(traj_data.atoms, c, energy_evaluator=traj_data.evaluator,
                         spin=1)
                for c in traj_data.final_trajectory
            ]
            expansions = [m.calculate_energy(order=2) for m in mols]
            grads = [e[1] for e in expansions]
            hessians = [e[2] for e in expansions]
            traj_data = traj_data._replace(
                final_gradients=grads,
                final_hessians=hessians
            )
        if traj_data.initial_ts_index is None:
            traj_data = traj_data._replace(
                initial_ts_index=DielsAlderReactionTrajectory.from_trajectory_data(traj_data, which='initial').ts_index
            )
        if traj_data.final_ts_index is None:
            traj_data = traj_data._replace(
                final_ts_index=DielsAlderReactionTrajectory.from_trajectory_data(traj_data, which='final').ts_index
            )

    else:
        traj_data:TrajectoryData
        if traj_data.gradients is None:
            mols = [
                Molecule(traj_data.atoms, c, energy_evaluator=traj_data.evaluator,
                         spin=1)
                for c in traj_data.coordinates
            ]
            expansions = [m.calculate_energy(order=2) for m in mols]
            grads = [e[1] for e in expansions]
            hessians = [e[2] for e in expansions]
            traj_data = traj_data._replace(
                gradients=grads,
                hessians=hessians
            )
        if traj_data.ts_index is None:
            traj_data = traj_data._replace(
                ts_index=DielsAlderReactionTrajectory.from_trajectory_data(traj_data).ts_index
            )
    return traj_data
# def reoptimize_trajectory(mol,
#                           init_traj,
#                           max_iterations=500,
#                           profile_generator='neb',
#                           energy_evaluator='aimnet2',
#                           **optimization_settings
#                           ):
#     base_structs = [
#         mol.modify(coords=t, energy_evaluator=energy_evaluator)
#         for t in init_traj
#     ]
#     init_engs, (ts, r, p) = get_critical_points(base_structs)
#     if p < r:
#         traj_structs = list(reversed(base_structs[p:r+1]))
#         traj_engs = list(reversed(init_engs[p:r+1]))
#     else:
#         traj_structs = base_structs[r:p+1]
#         traj_engs = init_engs[r:p+1]
#
#     eeee = Reaction([traj_structs[0]], [traj_structs[-1]],
#                     profile_generator=profile_generator,
#                     energy_evaluator=energy_evaluator
#                     )
#     prof = eeee.get_profile_generator(
#         energy_evaluator=energy_evaluator
#     )
#     new_geoms = prof.generate(
#         base_images=traj_structs,
#         max_iterations=max_iterations,
#         **optimization_settings
#     )
#
#     new_rmsds = prof.evaluate_profile_distances(new_geoms, normalize=False)
#     new_engs = prof.evaluate_profile_energies(new_geoms)
#     old_rmsds = prof.evaluate_profile_distances(traj_structs, normalize=False)
#
#     return ReoptimizedTrajectoryData(
#         atoms=base_structs[0].atoms,
#         final_trajectory=np.array([t.coords for t in new_geoms]),
#         final_energies=new_engs,
#         final_rmsds=new_rmsds,
#         initial_trajectory=np.array([t.coords for t in traj_structs]),
#         initial_energies=traj_engs,
#         initial_rmsds=old_rmsds,
#         raw_pre_sampling=init_traj,
#         raw_pre_energies=init_engs,
#         evaluator=energy_evaluator
#     )

CriticalPointIndices = collections.namedtuple('CriticalPointIndices',
                                              ['ts', 'react', 'prod'])
def get_critical_points(trajectory, energies=None, initial=None, gradients=None,
                        hessians=None,
                        small_freq_cutoff=5e-4,  # roughly 100 cm-1
                        large_freq_cutoff=5e-3,  # roughly 1000 cm-1
                        ts_idx=None,
                        reactant_idx=None):
    if energies is None:
        energies = [g.calculate_energy() for g in trajectory]

    product_idx = np.argmin(energies)
    if ts_idx is None:
        if trajectory is not None and hessians is not None:
            freqs = [
                g.modify(potential_derivatives=[0, h]).get_normal_modes().freqs
                for g,h in zip(trajectory, hessians)
            ]

            choices = [
                i for i,f in enumerate(freqs)
                if np.sum((f < 0) & (np.abs(f) > small_freq_cutoff) & (np.abs(f) < large_freq_cutoff)) == 1
            ]
            if len(choices) == 0:
                ts_idx = np.argmax(energies)
            else:
                subidx = np.argmax([energies[c] for c in choices])
                ts_idx = choices[subidx]
        else:
            ts_idx = np.argmax(energies)
    if reactant_idx is None:
        if product_idx > ts_idx:
            react_idx = np.argmin(energies[:ts_idx])
        else:
            react_idx = np.argmin(energies[ts_idx:]) + ts_idx
    else:
        react_idx = reactant_idx
    return energies, CriticalPointIndices(ts_idx, react_idx, product_idx)

def refine_trajectory(product_data: ReoptimizedTrajectoryData | TrajectoryData,
                      trajectory_data: ReoptimizedTrajectoryData | TrajectoryData = None,
                      energy_evaluator=None,
                      profile_generator='neb',
                      output_dir=None,
                      info_file='refined.json',
                      method_options=None,
                      climb=True,
                      ts_opt_generator='pys-ts',  # 'pys-dimer',
                      ts_opt_settings=None,
                      ts_opt_optimizer=None,
                      post_opt_generator='neb',  # 'pys-dimer',
                      post_opt_settings=None,
                      post_opt_optimizer=None,
                      # thresh='gau_tight',
                      thresh=None,
                      tol=None,
                      max_displacement=None,
                      refine_endpoints=True,
                      refine_ts=True,
                      optimizer='pysis',
                      optimizer_method='rfo',
                      optimizer_settings=None,
                      which='final',
                      max_iterations=500,
                      max_refinement_iterations=3,
                      fix_ts=True,
                      logger=None,
                      **calc_options
                      ):
    if trajectory_data is None:
        trajectory_data = product_data
    # init_js = dev.read_json(TestManager.test_data('product.json'))
    # new_js = dev.read_json(TestManager.test_data('trajectory.json'))
    if energy_evaluator is None:
        energy_evaluator = product_data.evaluator

    if optimizer_settings is None:
        optimizer_settings = {}

    for k,v in dict(
            thresh=thresh,
            tol=tol,
            max_displacement=max_displacement,
    ).items():
        if v is not None and k not in optimizer_settings:
            optimizer_settings[k] = v

    if hasattr(trajectory_data, 'final_trajectory'):
        trajectory_data: ReoptimizedTrajectoryData
        if which == 'initial':
            trajectory = trajectory_data.initial_trajectory
            energies = trajectory_data.initial_energies
            grads = trajectory_data.initial_gradients
            hess = trajectory_data.initial_hessians
            ts_idx = trajectory_data.initial_ts_index
            rmsds = trajectory_data.initial_rmsds
            raw_pre_sampling = None
            raw_pre_energies = None
        else:
            trajectory = trajectory_data.final_trajectory
            energies = trajectory_data.final_energies
            grads = trajectory_data.final_gradients
            hess = trajectory_data.final_hessians
            ts_idx = trajectory_data.final_ts_index
            rmsds = trajectory_data.final_rmsds
            raw_pre_sampling = trajectory_data.raw_pre_sampling
            raw_pre_energies = trajectory_data.raw_pre_energies
    else:
        trajectory_data: TrajectoryData
        trajectory = trajectory_data.coordinates
        energies = trajectory_data.energies
        grads = trajectory_data.gradients
        hess = trajectory_data.hessians
        ts_idx = trajectory_data.ts_index
        rmsds = trajectory_data.rmsds
        raw_pre_sampling = None
        raw_pre_energies = None

    traj = [
        Molecule(product_data.atoms,
                 c,
                 energy_evaluator=energy_evaluator,
                 spin=1)
        for c in trajectory
    ]

    _, inds = get_critical_points(traj, energies=energies, hessians=hess, ts_idx=ts_idx)
    if refine_endpoints:
        # uh = traj[0]
        traj[inds.react] = traj[inds.react].optimize(
            mode=optimizer,
            method=optimizer_method,
            max_iterations=max_iterations, logger=logger)
        # print(traj[0].calculate_energy() - uh.calculate_energy())
        # uh2 = traj[-1]
        traj[inds.prod] = traj[inds.prod].optimize(
            mode=optimizer,
            method=optimizer_method,
            max_iterations=max_iterations, logger=logger)
        # print(traj[-1].calculate_energy() - uh2.calculate_energy())


    if refine_ts:
        if ts_opt_generator is not None:
            if ts_opt_settings is None:
                ts_opt_settings = optimizer_settings | dict(optimizer=ts_opt_optimizer)
            if 'max_iterations' not in ts_opt_settings:
                ts_opt_settings['max_iterations'] = max_iterations
            elif max_refinement_iterations is None:
                max_refinement_iterations = max_iterations
        elif max_refinement_iterations is None:
            max_refinement_iterations = max_iterations
        ts_opt_settings = dict(
            logger=logger
        ) | ts_opt_settings

        rxn = Reaction([traj[inds.react]], [traj[inds.prod]])
        if method_options is None:
            method_options = {}
        prof = rxn.get_profile_generator(profile_generator,
                                         energy_evaluator=energy_evaluator,
                                         climb=climb,
                                         **method_options)
        if fix_ts:
            calc_options['fixed_images'] = [inds.ts]
        new_images = prof.generate(base_images=traj,
                                   optimizer_settings=optimizer_settings,
                                   logger=logger,
                                   max_iterations=max_refinement_iterations,
                                   **calc_options)
        expansions = [i.calculate_energy(order=2) for i in new_images]
        new_energies = [e[0] for e in expansions]
        new_grads = [e[1] for e in expansions]
        new_hessians = [e[2] for e in expansions]

        _, subinds = get_critical_points(new_images, energies=new_energies, hessians=new_hessians,
                                         ts_idx=(inds.ts if fix_ts and len(new_images) == len(traj) else None))
        if ts_opt_generator is not None:
            rxn = Reaction([new_images[subinds.react]], [new_images[subinds.prod]])
            prof = rxn.get_profile_generator(ts_opt_generator,
                                             energy_evaluator=energy_evaluator,
                                             climb=climb,
                                             **method_options)
            new_images2 = prof.generate(base_images=new_images,
                                       **ts_opt_settings)

            mod_pos = [
                (i,m) for i,m in enumerate(new_images2)
                if m not in new_images
            ]
            new_expansions = [(i,m.calculate_energy(order=2)) for i,m in mod_pos]
            for i,e in new_expansions:
                new_energies[i] = e[0]
                new_grads[i] = e[1]
                new_hessians[i] = e[2]
            new_images = new_images2
            _, subinds = get_critical_points(new_images, energies=new_energies, hessians=new_hessians,
                                             ts_idx=(subinds.ts if fix_ts and len(new_images) == len(traj) else None))

            if post_opt_generator is not None:
                if post_opt_settings is None:
                    post_opt_settings = optimizer_settings
                post_opt_settings = dict(
                    logger=logger,
                    max_iterations=max_refinement_iterations,
                ) | post_opt_settings

                if fix_ts:
                    post_opt_settings['fixed_images'] = [inds.ts]

                rxn = Reaction([new_images[subinds.react]], [new_images[subinds.prod]])
                prof = rxn.get_profile_generator(post_opt_generator,
                                                 energy_evaluator=energy_evaluator,
                                                 climb=climb,
                                                 **method_options)
                new_images2 = prof.generate(base_images=new_images,
                                            **post_opt_settings)

                mod_pos = [
                    (i, m) for i, m in enumerate(new_images2)
                    if m not in new_images
                ]
                new_expansions = [(i, m.calculate_energy(order=2)) for i, m in mod_pos]
                for i, e in new_expansions:
                    new_energies[i] = e[0]
                    new_grads[i] = e[1]
                    new_hessians[i] = e[2]
                new_images = new_images2
                _, subinds = get_critical_points(new_images, energies=new_energies, hessians=new_hessians,
                                                 ts_idx=(subinds.ts if fix_ts and len(new_images) == len(traj) else None))

        new_ts_idx = subinds.ts
    else:
        new_images = traj
        new_energies = energies
        new_grads = grads
        new_hessians = hess
        new_ts_idx = ts_idx

    new_coords = np.array([t.coords for t in new_images])
    new_traj = ReoptimizedTrajectoryData(
        atoms=product_data.atoms,
        final_trajectory=np.array([t.coords for t in new_images]),
        final_energies=new_energies,
        final_gradients=new_grads,
        final_hessians=new_hessians,
        final_ts_index=new_ts_idx,
        final_rmsds=nput.incremental_eckart_rmsd(new_coords, masses=new_images[0].masses, mass_weighted=False),
        initial_trajectory=trajectory,
        initial_energies=energies,
        initial_gradients=grads,
        initial_hessians=hess,
        initial_ts_index=ts_idx,
        initial_rmsds=rmsds,
        raw_pre_sampling=raw_pre_sampling,
        raw_pre_energies=raw_pre_energies,
        evaluator=energy_evaluator
    )

    if output_dir is not None:
        utils.write_namedtuple(
            os.path.join(output_dir, info_file),
            new_traj
        )

    return new_traj

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
        ts_idx=None,
        reactant_idx=None,
        mark_critical_points=True,
        critical_point_style=None,
        baseline=None,
        hessians=None,
        **opts):
    energies, (ts, r, p) = get_critical_points(None, energies,
                                               ts_idx=ts_idx,
                                               hessians=hessians,
                                               reactant_idx=reactant_idx)
    energies = np.asanyarray(energies)
    if distance_metric is None:
        distance_metric = cc_single_normalized_distance
    if metric_label is None:
        metric_label = metric_label_map.get(distance_metric)
        if metric_label is None:
            metric_label = distance_metric.__name__
    x1 = distance_metric(coordinates, bonds) * UnitsData.convert("BohrRadius", "Angstroms")
    if baseline is None:
        baseline = energies[r]
    e = (energies - baseline) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
    figure = plt.Plot(
        x1,
        e,
        **(dict(
            axes_labels=[
                metric_label + r" ($\AA$)",
                "E (kcal mol$^{-1}$)"
            ]
        ) | opts)
    )
    if mark_critical_points:
        if critical_point_style is None:
            critical_point_style = {'color':'black'}
        plt.ScatterPlot(
            [x1[r], x1[ts], x1[p]],
            [e[r], e[ts], e[p]],
            **(dict(figure=figure) | critical_point_style)
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
                 hessians=None,
                 gradients=None,
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
        self._gradients = gradients
        self._hessians = hessians
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
                energy_evaluator=self.energy_evaluator,
                spin=1
            )
        return self._mols[i]

    @property
    def energies(self):
        if self._energies is None:
            self._energies, self._gradients, self._hessians = self._load_expansions()
        return self._energies
    @property
    def gradients(self):
        if self._gradients is None:
            self._energies, self._gradients, self._hessians = self._load_expansions()
        return self._gradients
    @property
    def hessians(self):
        if self._hessians is None:
            self._energies, self._gradients, self._hessians = self._load_expansions()
        return self._hessians
    def _load_expansions(self):
        expansions = [g.calculate_energy(order=2) for g in self.mols]
        return [e[0] for e in expansions], [e[1] for e in expansions], [e[2] for e in expansions]

    @property
    def mols(self):
        for i in range(len(self.structures)):
            self.load_mol(i)
        return self._mols

    @property
    def ts_index(self):
        if self._ts_idx is None:
            _, (self._ts_idx, self._reactant_idx, self._product_idx) = get_critical_points(self.mols,
                                                                                           energies=self.energies,
                                                                                           hessians=self._hessians,
                                                                                           ts_idx=self._ts_idx)
        return self._ts_idx
    @property
    def reactant_index(self):
        if self._reactant_idx is None:
            _, (self._ts_idx, self._reactant_idx, self._product_idx) = get_critical_points(self.mols,
                                                                                           energies=self.energies,
                                                                                           hessians=self._hessians,
                                                                                           ts_idx=self._ts_idx)
        return self._reactant_idx
    @property
    def product_index(self):
        if self._product_idx is None:
            _, (self._ts_idx, self._reactant_idx, self._product_idx) = get_critical_points(self.mols,
                                                                                           energies=self.energies,
                                                                                           hessians=self._hessians,
                                                                                           ts_idx=self._ts_idx)
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
                             gradients=None,
                             hessians=None,
                             energy_evaluator=None,
                             ts_index=None,
                             which='final',
                             **etc):
        if isinstance(trajectory_data, str):
            trajectory_data = utils.read_namedtuple(trajectory_data, ReoptimizedTrajectoryData)
        elif isinstance(trajectory_data, dict):
            if 'trajectory_energies' in trajectory_data:
                trajectory_data = TrajectoryData(
                    atoms=trajectory_data['atoms'],
                    coordinates=trajectory_data['trajectory'],
                    energies=trajectory_data['trajectory_energies'],
                    gradients=trajectory_data['trajectory_gradients'],
                    hessians=trajectory_data['trajectory_hessians'],
                    rmsds=None,
                    ts_index=trajectory_data['ts_index'],
                    evaluator=energy_evaluator
                )
            elif 'initial_trajectory_energies' in trajectory_data:
                trajectory_data = ReoptimizedTrajectoryData(
                    atoms=trajectory_data['atoms'],
                    initial_trajectory=trajectory_data['initial_trajectory'],
                    initial_energies=trajectory_data['initial_trajectory_energies'],
                    initial_gradients=trajectory_data['initial_trajectory_gradients'],
                    initial_hessians=trajectory_data['initial_trajectory_hessians'],
                    initial_ts_index=trajectory_data['initial_ts_index'],
                    final_trajectory=trajectory_data['refined_trajectory'],
                    final_energies=trajectory_data['refined_trajectory_energies'],
                    final_gradients=trajectory_data['refined_trajectory_gradients'],
                    final_hessians=trajectory_data['refined_trajectory_hessians'],
                    final_ts_index=trajectory_data['refined_ts_index'],
                    initial_rmsds=None,
                    final_rmsds=None,
                    raw_pre_energies=None,
                    raw_pre_sampling=None,
                    evaluator=energy_evaluator
                )
            elif 'final_energies' in trajectory_data:
                trajectory_data = ReoptimizedTrajectoryData(**trajectory_data)
            else:
                trajectory_data = TrajectoryData(**trajectory_data)

        if energies is None:
            if hasattr(trajectory_data, 'final_energies'):
                trajectory_data: ReoptimizedTrajectoryData
                if which == 'final':
                    energies = trajectory_data.final_energies
                    gradients = trajectory_data.final_gradients
                    hessians = trajectory_data.final_hessians
                else:
                    energies = trajectory_data.initial_energies
                    gradients = trajectory_data.initial_gradients
                    hessians = trajectory_data.initial_hessians
            else:
                trajectory_data: TrajectoryData
                energies = trajectory_data.energies
                gradients = trajectory_data.gradients
                hessians = trajectory_data.hessians

        if structures is None:
            if hasattr(trajectory_data, 'final_energies'):
                if which == 'final':
                    structures = trajectory_data.final_trajectory
                else:
                    structures = trajectory_data.initial_trajectory
            else:
                structures = trajectory_data.coordinates

        if ts_index is None:
            if hasattr(trajectory_data, 'final_energies'):
                if which == 'final':
                    ts_index = trajectory_data.final_ts_index
                else:
                    ts_index = trajectory_data.initial_ts_index
            else:
                ts_index = trajectory_data.ts_index

        if energy_evaluator is None:
            energy_evaluator = trajectory_data.evaluator
        return cls(
            trajectory_data.atoms,
            structures=structures,
            energies=energies,
            gradients=gradients,
            hessians=hessians,
            energy_evaluator=energy_evaluator,
            ts_index=ts_index,
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
        data = write_trajectory(
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
                     **opts):
        return plot_reaction_profile(
            self.structures,
            self.energies,
            distance_metric=distance_metric,
            metric_label=metric_label,
            bonds=bonds,
            return_metrics=return_metrics,
            ts_idx=self.ts_index,
            reactant_idx=self.reactant_index,
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
                         baseline=None,
                         **opts):
        if baseline is None:
            baseline = self.energies[self.reactant_index]
        figure, metrics = self.plot_profile(
            distance_metric=distance_metric,
            metric_label=metric_label,
            bonds=bonds,
            figure=figure,
            return_metrics=True,
            baseline=baseline,
            **(opts | dict(label=labels[0] if labels is not None else None))
        )
        if comparison_styles is None:
            comparison_styles = {'linestyle':'dashed'}
        other.plot_profile(
            distance_metric=distance_metric,
            metric_label=metric_label,
            bonds=bonds,
            figure=figure,
            baseline=baseline,
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
