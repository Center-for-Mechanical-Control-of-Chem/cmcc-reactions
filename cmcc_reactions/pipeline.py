from __future__ import annotations

import tempfile
import glob

import numpy as np
from dataclasses import dataclass
import collections
import os

from McUtils.Data import UnitsData
import McUtils.Plots as plt
import McUtils.Devutils as dev
import McUtils.Coordinerds as coordops
import McUtils.Iterators as itut
import McUtils.Numputils as nput
from McUtils.ExternalPrograms import sbatch_python_job

from . import utils
from . import generate_reaction_products as gen_prods
from . import optimal_directions as fopt
from . import trajectory_tools as rda


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

OptimizedForcePipelineData = collections.namedtuple(
    'OptimizedForcePipelineData',
    [
        'smiles',
        'atoms',
        'bonds',
        'breakpoints',
        "energy_evaluator",
        'product_geometry',
        'product_energy',
        'product_optimization_settings',
        'initial_trajectory',
        'initial_trajectory_energies',
        'initial_trajectory_gradients',
        'initial_trajectory_hessians',
        'initial_trajectory_ts_index',
        'refined_trajectory',
        'refined_trajectory_energies',
        'refined_trajectory_gradients',
        'refined_trajectory_hessians',
        'refined_trajectory_ts_index',
        'trajectory_optimization_settings',
        'reactant_geometry',
        'reactant_energy',
        'transition_state_geometry',
        'transition_state_energy',
        'reactant_hessian',
        'transition_state_hessian',
        'force_coeffs',
        'internals',
        'use_mode_space',
        'force_optimization_settings',
        'force_modified_reactant_geometries',
        'force_modified_reactant_energies',
        'force_modified_transition_state_geometries',
        'force_modified_transition_state_energies',
        'force_vectors',
        'force_magnitudes',
        'force_units',
        'mass_weight',
        'force_optimizer_settings'
    ]
)
utils.register_namedtuple(OptimizedForcePipelineData,
                          defaults={
                              'initial_trajectory_gradients': None,
                              'initial_trajectory_hessians': None,
                              'initial_trajectory_ts_index': None,
                              'refined_trajectory_gradients': None,
                              'refined_trajectory_hessians': None,
                              'refined_trajectory_ts_index': None,
                          }
                          )

@dataclass
class OptimizedForceResults:
    product: gen_prods.InitialProductData|None = None
    trajectory: gen_prods.ReoptimizedTrajectoryData|None = None
    optimized_forces: fopt.OptimizedForceData|None = None
    fmrds: list[fopt.ForceModifiedReactionData]|None = None

    _optimizer = None

    field_mapping = {
        'product':
            {
                'smiles':'smiles',
                'atoms':'atoms',
                'bonds':'bonds',
                'breakpoints':'breakpoints',
                'product_geometry':'coords',
                'product_energy':'energy',
                'energy_evaluator':'evaluator',
                'product_optimization_settings':'optimization_settings'
            },
        'trajectory':
            {
                'initial_trajectory':'initial_trajectory',
                'initial_trajectory_energies':'initial_energies',
                'initial_trajectory_gradients':'initial_gradients',
                'initial_trajectory_hessians':'initial_hessians',
                'initial_trajectory_ts_index':'initial_ts_index',
                'refined_trajectory':'final_trajectory',
                'refined_trajectory_energies':'final_energies',
                'refined_trajectory_gradients':'final_gradients',
                'refined_trajectory_hessians':'final_hessians',
                'refined_trajectory_ts_index':'final_ts_index',
                'trajectory_optimization_settings':'optimization_settings'
            },
        'optimized_forces':
            dict(
                reactant_geometry='reactant_geom',
                transition_state_geometry='transition_state_geom',
                reactant_hessian='reactant_hessian',
                transition_state_hessian='transition_state_hessian',
                force_coeffs='force_coeffs',
                internals='internals',
                use_mode_space='use_mode_space',
                force_optimization_settings='optimizer_settings'
            ),
        'force_modified_reactions': {
            'global': {
                'force_units': 'force_units',
                'mass_weight': 'mass_weight',
                'force_optimizer_settings': 'optimizer_settings',
                'reactant_energy': 'reactant_energy',
                'transition_state_energy': 'transition_state_energy'
            },
            'instance': {
                'force_modified_reactant_geometries': 'force_modified_reactant_geom',
                'force_modified_reactant_energies': 'force_modified_reactant_energy',
                'force_modified_transition_state_geometries': 'force_modified_transition_state_geom',
                'force_modified_transition_state_energies': 'force_modified_transition_state_energy',
                'force_vectors': 'force_vector',
                'force_magnitudes': 'force_magnitude'
            }
        }
    }
    def to_data(self) -> OptimizedForcePipelineData:
        opts = {}
        if self.product is not None:
            for field, key in self.field_mapping['product'].items():
                opts[field] = getattr(self.product, key)
        else:
            for field, key in self.field_mapping['product'].items():
                opts[field] = None

        if self.trajectory is not None:
            if opts['atoms'] is None:
                opts['atoms'] = self.trajectory.atoms
            if opts['energy_evaluator'] is None:
                opts['energy_evaluator'] = self.trajectory.evaluator
            for field, key in self.field_mapping['trajectory'].items():
                opts[field] = getattr(self.trajectory, key)
        else:
            for field, key in self.field_mapping['trajectory'].items():
                opts[field] = None

        if self.optimized_forces is not None:
            if opts['atoms'] is None:
                opts['atoms'] = self.optimized_forces.atoms
            if opts['energy_evaluator'] is None:
                opts['energy_evaluator'] = self.optimized_forces.energy_evaluator
            for field, key in self.field_mapping['optimized_forces'].items():
                opts[field] = getattr(self.optimized_forces, key)
        else:
            for field, key in self.field_mapping['optimized_forces'].items():
                opts[field] = None

        if self.fmrds is not None and len(self.fmrds) > 0:
            for k,v in {
                'atoms':'atoms',
                'internals':'internals',
                'reactant_geometry':'reactant_geom',
                'transition_state_geometry':'transition_state_geom',
                'energy_evaluator':'energy_evaluator'
            }.items():
                if opts[k] is None:
                    opts[k] = getattr(self.fmrds[0], v)
            for field, key in self.field_mapping['force_modified_reactions']['global'].items():
                opts[field] = getattr(self.fmrds[0], key)
            for field, key in self.field_mapping['force_modified_reactions']['instance'].items():
                opts[field] = [getattr(f, key) for f in self.fmrds]
        else:
            for field, key in self.field_mapping['force_modified_reactions']['global'].items():
                opts[field] = None
            for field, key in self.field_mapping['force_modified_reactions']['instance'].items():
                opts[field] = None

        return OptimizedForcePipelineData(**opts)

    @classmethod
    def from_data(cls, data: OptimizedForcePipelineData):
        if data.product_geometry is not None:
            product = gen_prods.InitialProductData(
                smiles=data.smiles,
                atoms=data.atoms,
                coords=data.product_geometry,
                bonds=data.bonds,
                energy=data.product_energy,
                breakpoints=data.breakpoints,
                evaluator=data.energy_evaluator,
                optimization_settings=data.product_optimization_settings,
            )
        else:
            product = None

        if data.refined_trajectory is not None:
            trajectory = gen_prods.ReoptimizedTrajectoryData(
                atoms=data.atoms,
                final_trajectory=data.refined_trajectory,
                final_energies=data.refined_trajectory_energies,
                final_gradients=data.refined_trajectory_gradients,
                final_hessians=data.refined_trajectory_hessians,
                final_ts_index=data.refined_trajectory_ts_index,
                final_rmsds=None,  # not stored in pipeline data
                initial_trajectory=data.initial_trajectory,
                initial_energies=data.initial_trajectory_energies,
                initial_gradients=data.initial_trajectory_gradients,
                initial_hessians=data.initial_trajectory_hessians,
                initial_ts_index=data.initial_trajectory_ts_index,
                initial_rmsds=None,  # not stored in pipeline data
                raw_pre_sampling=None,  # not stored in pipeline data
                raw_pre_energies=None,  # not stored in pipeline data
                evaluator=data.energy_evaluator,
                optimization_settings=data.trajectory_optimization_settings,
            )
        else:
            trajectory = None

        if data.force_coeffs is not None:
            optimized_forces = fopt.OptimizedForceData(
                atoms=data.atoms,
                reactant_geom=data.reactant_geometry,
                transition_state_geom=data.transition_state_geometry,
                reactant_hessian=data.reactant_hessian,
                transition_state_hessian=data.transition_state_hessian,
                energy_evaluator=data.energy_evaluator,
                force_coeffs=data.force_coeffs,
                internals=data.internals,
                use_mode_space=data.use_mode_space,
                optimizer_settings=data.force_optimization_settings
            )
        else:
            optimized_forces = None

        if data.force_modified_reactant_geometries is not None:
            fmrds = [
                fopt.ForceModifiedReactionData(
                    atoms=data.atoms,
                    reactant_geom=data.reactant_geometry,
                    reactant_energy=data.reactant_energy,
                    transition_state_geom=data.transition_state_geometry,
                    transition_state_energy=data.transition_state_energy,
                    force_modified_reactant_geom=fmr_geom,
                    force_modified_reactant_energy=fmr_energy,
                    force_modified_transition_state_geom=fmt_geom,
                    force_modified_transition_state_energy=fmt_energy,
                    force_vector=fv,
                    force_magnitude=fm,
                    force_units=data.force_units,
                    mass_weight=data.mass_weight,
                    energy_evaluator=data.energy_evaluator,
                    internals=data.internals,
                    optimizer_settings=data.force_optimizer_settings,
                )
                for fmr_geom, fmr_energy, fmt_geom, fmt_energy, fv, fm in zip(
                    data.force_modified_reactant_geometries,
                    data.force_modified_reactant_energies,
                    data.force_modified_transition_state_geometries,
                    data.force_modified_transition_state_energies,
                    data.force_vectors,
                    data.force_magnitudes,
                )
            ]
        else:
            fmrds = None

        return cls(
            product=product,
            trajectory=trajectory,
            optimized_forces=optimized_forces,
            fmrds=fmrds
        )

    @classmethod
    def from_intermediate_data(cls, subdata) -> OptimizedForceResults:
        if isinstance(subdata, dict):
            subdata = utils.make_namedtuple(subdata)
        if isinstance(subdata, OptimizedForceResults):
            return subdata
        elif utils.isnamedtupleinstance(subdata, OptimizedForcePipelineData):
            return cls.from_data(subdata)
        elif utils.isnamedtupleinstance(subdata, gen_prods.InitialProductData):
            return cls(product=subdata)
        elif utils.isnamedtupleinstance(subdata, gen_prods.ReoptimizedTrajectoryData):
            return cls(trajectory=subdata)
        elif utils.isnamedtupleinstance(subdata, fopt.OptimizedForceData):
            return cls(optimized_forces=subdata)
        elif utils.isnamedtupleinstance(subdata, fopt.ForceModifiedReactionData):
            return cls(fmrds=[subdata])
        else:
            raise ValueError(f"can't construct from intermediate data {subdata}")

    @classmethod
    def from_file(cls, file) -> OptimizedForceResults:
        return cls.from_intermediate_data(utils.read_namedtuple(file))
    def save(self, output_dir, info_file='pipeline_data.json'):
        if os.path.splitext(output_dir)[-1].startswith('.'):
            output_dir, info_file = os.path.split(output_dir)
        if len(output_dir) > 0:
            os.makedirs(output_dir, exist_ok=True)
            info_file = os.path.join(output_dir, info_file)
        traj_data = self.to_data()
        utils.write_namedtuple(
            info_file,
            traj_data
        )
        return info_file
    @property
    def optimizer(self):
        if self.optimized_forces is None:
            return None
        else:
            if self._optimizer is None:
                self._optimizer = fopt.ForceOptimizer.from_data(self.optimized_forces)
            return self._optimizer

    def trajectory_analyzer(self, **opts):
        return rda.DielsAlderReactionTrajectory.from_trajectory_data(self.trajectory, **opts)

    def plot_fmrd_lines(self,
                        fmrd_index,
                        fmrd=None,
                        bar_color='gray',
                        bar_spacing=.2,
                        distance_metric=None,
                        force_modified=True,
                        bonds=((0, 2), (1, 3)),
                        baseline=None,
                        traj=None,
                        **etc
                        ):
        distance_metric = rda.resolve_distance_metric(distance_metric)
        if fmrd is None:
            fmrd: fopt.ForceModifiedReactionData = self.fmrds[fmrd_index]
        if traj is None:
            traj = self.trajectory_analyzer()
        product_energy = traj.energies[traj.product_index]
        product_coords = traj.product.coords
        if force_modified:
            coords = distance_metric(
                [fmrd.force_modified_reactant_geom, fmrd.force_modified_transition_state_geom, product_coords],
                bonds
            ) * UnitsData.convert("BohrRadius", "Angstroms")
            engs = [
                fmrd.force_modified_reactant_energy,
                fmrd.force_modified_transition_state_energy,
                product_energy
            ]
        else:
            coords = distance_metric(
                [fmrd.reactant_geom, fmrd.transition_state_geom, product_coords],
                bonds
            ) * UnitsData.convert("BohrRadius", "Angstroms")
            engs = [
                fmrd.reactant_energy,
                fmrd.transition_state_energy,
                product_energy
            ]

        if baseline is None:
            if traj is not None:
                baseline = traj.energies[traj.reactant_index]
            else:
                baseline = fmrd.reactant_energy
        return rda.plot_reaction_lines(
            coords,
            engs,
            connect=True,
            ticks=False,
            baseline=baseline,
            color=bar_color,
            bar_spacing=bar_spacing,
            **etc
        )
    def plot_profile(self,
                     fmrd_index=None,
                     fmrd=None,
                     distance_metric=None,
                     bonds=((0, 2), (1, 3)),
                     bar_color='gray',
                     bar_spacing=.2,
                     which='initial',
                     force_modified='both',
                     **opts):
        traj = rda.DielsAlderReactionTrajectory.from_trajectory_data(self.trajectory, which=which)
        figure, x = traj.plot_profile(
            return_metrics=True,
            distance_metric=distance_metric,
            **opts
        )
        if force_modified and fmrd_index is None and fmrd is None:
            fmrd_index = 0
        if fmrd_index is not None or fmrd is not None:
            if dev.str_is(force_modified, 'both'):
                if isinstance(bar_color, str):
                    bar_color = [
                        plt.prep_color(bar_color, lighten=.5),
                        bar_color
                    ]
                self.plot_fmrd_lines(
                    fmrd_index,
                    fmrd=fmrd,
                    distance_metric=distance_metric,
                    bonds=bonds,
                    bar_color=bar_color[0],
                    bar_spacing=bar_spacing,
                    figure=figure,
                    force_modified=False,
                    traj=traj,
                )
                self.plot_fmrd_lines(
                    fmrd_index,
                    fmrd=fmrd,
                    distance_metric=distance_metric,
                    bonds=bonds,
                    bar_color=bar_color[1],
                    bar_spacing=bar_spacing,
                    figure=figure,
                    force_modified=True,
                    traj=traj
                )
            else:
                self.plot_fmrd_lines(
                    fmrd_index,
                    fmrd=fmrd,
                    distance_metric=distance_metric,
                    bonds=bonds,
                    bar_color=bar_color,
                    bar_spacing=bar_spacing,
                    figure=figure,
                    force_modified=force_modified,
                    traj=traj
                )
        return figure
    def compare_profiles(self,
                         fmrd_index=None,
                         distance_metric=None, bonds=((0, 2), (1, 3)),
                         bar_color='gray',
                         bar_spacing=.2,
                         force_modified='both',
                         fmrd=None,
                         **opts):
        traj = rda.DielsAlderReactionTrajectory.from_trajectory_data(self.trajectory)
        figure, x = traj.compare_profiles(
            rda.DielsAlderReactionTrajectory.from_trajectory_data(self.trajectory, which='initial'),
            return_metrics=True,
            distance_metric=distance_metric,
            **opts
        )
        if force_modified and fmrd_index is None and fmrd is None:
            fmrd_index = 0
        if fmrd_index is not None or fmrd is not None:
            if dev.str_is(force_modified, 'both'):
                if isinstance(bar_color, str):
                    bar_color = [
                        plt.prep_color(bar_color, lighten=.5),
                        bar_color
                    ]
                self.plot_fmrd_lines(
                    fmrd_index,
                    distance_metric=distance_metric,
                    bonds=bonds,
                    bar_color=bar_color[0],
                    bar_spacing=bar_spacing,
                    figure=figure,
                    force_modified=False,
                    traj=traj,
                    fmrd=fmrd
                )
                self.plot_fmrd_lines(
                    fmrd_index,
                    distance_metric=distance_metric,
                    bonds=bonds,
                    bar_color=bar_color[1],
                    bar_spacing=bar_spacing,
                    figure=figure,
                    force_modified=True,
                    traj=traj,
                    fmrd=fmrd
                )
            else:
                self.plot_fmrd_lines(
                    fmrd_index,
                    distance_metric=distance_metric,
                    bonds=bonds,
                    bar_color=bar_color,
                    bar_spacing=bar_spacing,
                    figure=figure,
                    force_modified=force_modified,
                    traj=traj,
                    fmrd=fmrd
                )
        return figure


    # product: gen_prods.InitialProductData|None = None
    # trajectory: gen_prods.ReoptimizedTrajectoryData|None = None
    # optimized_forces: fopt.OptimizedForceData|None = None
    # fmrds: list[fopt.ForceModifiedReactionData]|None = None

def run_initial_sampling(product, output_dir=None, **opts):
    return gen_prods.generate_reactants_from_products(
        product,
        output_dir=output_dir,
        **opts
    )

def run_refined_trajectory(traj, output_dir=None, **opts):
    return rda.refine_trajectory(
        traj,
        output_dir=output_dir,
        **opts
    )

def run_update_trajectory(traj, **opts):
    return rda.update_trajectory_data(traj, **opts)

def run_force_optimization(trajectory,
                           internals='auto',
                           breakpoints=((0, 2), (1, 3)),
                           fragment_indices=1,
                           fix_breakpoint_atoms=True,
                           projection_internals='auto',
                           remove_fragment_transrot=True,
                           remove_local_transrot=True,
                           allow_mode_mixing=True,
                           which='final',
                           **opts):
    trajectory = rda.DielsAlderReactionTrajectory.from_trajectory_data(trajectory, which=which)
    ref = trajectory.reactant
    if dev.str_is(internals, 'auto'):
        if breakpoints is not None:
            inds = ref.fragment_indices
            internals = ref.get_bond_zmatrix(
                connect_fragments=True,
                fragment_ordering=list(range(len(inds))),
                attachment_points={breakpoints[0][0]: breakpoints[0][1]}
            )

    if dev.str_is(projection_internals, 'auto'):
        inds = ref.fragment_indices
        projection_internals = [
            coordops.extract_zmatrix_internals(z)
            for z in ref.get_bond_zmatrix(connect_fragments=False,
                                          fragment_ordering=list(range(len(inds))),
                                          attachment_points={breakpoints[0][0]: breakpoints[0][1]}
                                          )
        ]
        if nput.is_int(fragment_indices):
            projection_internals = projection_internals[fragment_indices]
        else:
            projection_internals = sum(projection_internals, [])

    if fix_breakpoint_atoms is not None:
        if nput.is_int(fragment_indices):
            fragment_indices = ref.fragment_indices[fragment_indices]
        fragment_indices = np.setdiff1d(fragment_indices, list(itut.flatten(breakpoints)))

    if projection_internals is not None and fragment_indices is not None:
        if nput.is_int(fragment_indices):
            fragment_indices = ref.fragment_indices[fragment_indices]
        projection_internals = [
            p for p in projection_internals
            if any(pp in fragment_indices for pp in p)
        ]

    opt = fopt.ForceOptimizer(trajectory.reactant, trajectory.transition_state,
                              fragment_indices=fragment_indices,
                              remove_fragment_transrot=remove_fragment_transrot,
                              remove_local_transrot=remove_local_transrot,
                              allow_mode_mixing=allow_mode_mixing,
                              projection_internals=projection_internals,
                              internals=internals,
                              **opts
                              )
    opt.optimize()
    return opt

def run_fmrds(optimizer,
              nmodes=15,
              magnitude=(-200, -100, -50, 50, 100, 200),
              pool=None,
              **opts):
    nmodes = min(optimizer.force_coeffs.shape[0], nmodes)
    return optimizer.reoptimize_with_force(
        list(range(nmodes)),
        magnitude=magnitude,
        pool=pool,
        **opts
    )

def run_internal_fmrds(optimizer,
                       internal_selector='dihedral',
                       magnitude=(-200, -100, -50, 50, 100, 200),
                       pool=None,
                       **opts):
    return optimizer.reoptimize_internals_with_force(
        internal_selector,
        magnitude=magnitude,
        pool=pool,
        **opts
    )

def run_optimization_pipeline(
        input_data: str | gen_prods.InitialProductData | OptimizedForceResults | OptimizedForcePipelineData,
        output_file=None,
        steps=None,
        verbose=False,
        trajectory_optimization_settings=None,
        refined_trajectory_optimization_settings=None,
        update_trajectory_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        internal_force_modification_settings=None,
        max_iterations=500,
        tol=1e-8,
        energy_evaluator=None,
        **global_options
) -> OptimizedForceResults:
    if isinstance(input_data, str):
        if output_file is None:
            output_file = input_data
        input_data = OptimizedForceResults.from_file(input_data)
    elif input_data is not None:
        input_data = OptimizedForceResults.from_intermediate_data(input_data)

    if isinstance(output_file, str) and os.path.isfile(output_file):
        if input_data is None:
            input_data = OptimizedForceResults.from_file(output_file)
        else:
            cache_data = OptimizedForceResults.from_file(output_file)
            input_data = OptimizedForceResults(
                product=input_data.product if input_data.product is not None else cache_data.product,
                trajectory=input_data.trajectory if input_data.trajectory is not None else cache_data.trajectory,
                optimized_forces=input_data.optimized_forces if input_data.optimized_forces is not None else cache_data.optimized_forces,
                fmrds=input_data.fmrds if input_data.fmrds is not None else cache_data.fmrds
            )
            # raise ValueError(f"got an existing `output_file` and `input_data` ({output_file} and {input_data})")

    input_data: OptimizedForceResults
    if steps is None:
        steps = []
        if input_data.fmrds is None:
            steps.append('fmrds')
            if input_data.optimized_forces is None:
                steps.append('optimized_forces')
                if input_data.trajectory is None:
                    steps.append('trajectory')
                    if input_data.product is None:
                        raise ValueError("product structure needed at minimum to run pipeline")
        steps = tuple(reversed(steps))

    product:gen_prods.InitialProductData = input_data.product
    if product is None:
        raise ValueError("product structure needed at minimum to run pipeline")

    if energy_evaluator is None:
        energy_evaluator = product.evaluator
    global_options = global_options | dict(
        max_iterations=max_iterations,
        tol=tol,
        energy_evaluator=energy_evaluator
    )
    try:
        if 'trajectory' in steps:
            if trajectory_optimization_settings is None:
                trajectory_optimization_settings = {}
            trajectory_optimization_settings = global_options | trajectory_optimization_settings
            if verbose:
                print('running sampling')
            input_data.trajectory = run_initial_sampling(product, **trajectory_optimization_settings)

            if output_file is not None:
                input_data.save(output_file)

        if 'update_trajectory' in steps:
            if update_trajectory_settings is None:
                update_trajectory_settings = {}
            # update_trajectory_settings = global_options | update_trajectory_settings
            if verbose:
                print('running updates')
            input_data.trajectory = run_update_trajectory(input_data.trajectory, **update_trajectory_settings)

            if output_file is not None:
                input_data.save(output_file)

        if 'refine' in steps:
            if refined_trajectory_optimization_settings is None:
                refined_trajectory_optimization_settings = {}
            refined_trajectory_optimization_settings = global_options | refined_trajectory_optimization_settings
            if verbose:
                print('running refinement')
            input_data.trajectory = run_refined_trajectory(input_data.trajectory, **refined_trajectory_optimization_settings)

            if output_file is not None:
                input_data.save(output_file)

        trajectory:gen_prods.ReoptimizedTrajectoryData = input_data.trajectory
        optimizer = None
        if 'optimized_forces' in steps:
            if verbose:
                print('running optimized forces')
            if trajectory is None:
                raise ValueError("trajectory needed to get optimized forces")

            if optimized_force_settings is None:
                optimized_force_settings = {}
            optimized_force_settings = global_options | optimized_force_settings
            optimized_force_settings.pop('energy_evaluator', None)
            optimizer = run_force_optimization(trajectory, **optimized_force_settings)
            input_data.optimized_forces = optimizer.to_data()

            if output_file is not None:
                input_data.save(output_file)

        if 'fmrds' in steps:
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running optimized fmrds')


            if force_modification_settings is None:
                force_modification_settings = {}
            force_modification_settings = global_options | force_modification_settings
            fmrds = run_fmrds(optimizer, **force_modification_settings)
            input_data.fmrds = [f[2] for f in fmrds]

            if output_file is not None:
                input_data.save(output_file)

        if 'internals' in steps:
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running internal forces')

            if internal_force_modification_settings is None:
                internal_force_modification_settings = force_modification_settings
            if internal_force_modification_settings is None:
                internal_force_modification_settings = {}
            internal_force_modification_settings = global_options | internal_force_modification_settings
            fmrds = run_internal_fmrds(optimizer, **internal_force_modification_settings)
            if input_data.fmrds is None:
                input_data.fmrds = [f[2] for f in fmrds]
            else:
                input_data.fmrds = input_data.fmrds + [f[2] for f in fmrds]

            if output_file is not None:
                input_data.save(output_file)
    finally:
        if output_file is not None:
            input_data.save(output_file)

    return input_data

def generate_from_product_library(
        template=None,
        fragments=None,
        active_sites=None,
        chiralities=None,
        output_dir=None,
        conf_gen_options=None,
        take_unique=True,
        num_structs=10,
        calc=None,
        evaluate_energy=True,
        energy_evaluator='aimnet2',
        preoptimize=True,
        optimizer_settings=None,
        smiles_hash_generator='inchi',
        parallelizer=None,
        batch_size=50,
        verbose=True,
        output_file="pipeline_data.json",
        steps=None,
        trajectory_optimization_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        max_iterations=500,
        tol=1e-8,
        submit=True,
        max_products=None,
        **global_options
):
    def callback(product_data, product_file):
        targ_dir = os.path.dirname(product_file)
        product_file = os.path.basename(product_file)
        curdir = os.getcwd()
        try:
            os.chdir(targ_dir)
            out_file = output_file
            if submit:
                script, _ = sbatch_python_job(
                    run_optimization_pipeline,
                    product_file,
                    out_file,
                    steps=steps,
                    trajectory_optimization_settings=trajectory_optimization_settings,
                    optimized_force_settings=optimized_force_settings,
                    force_modification_settings=force_modification_settings,
                    max_iterations=max_iterations,
                    tol=tol,
                    energy_evaluator=energy_evaluator,
                    verbose=verbose,
                    sbatch_kwargs={}, # disable stuff
                    post_processor=None,
                    **global_options
                )
                if verbose:
                    print(script.run())
            else:
                run_optimization_pipeline(
                    product_data,
                    out_file,
                    steps=steps,
                    trajectory_optimization_settings=trajectory_optimization_settings,
                    optimized_force_settings=optimized_force_settings,
                    force_modification_settings=force_modification_settings,
                    max_iterations=max_iterations,
                    tol=tol,
                    energy_evaluator=energy_evaluator,
                    verbose=verbose,
                    **global_options
                )
                print(f"Wrote to `{out_file}`")
        finally:
            os.chdir(curdir)

    if steps is None or 'products' in steps:
        gen_prods.generate_products_and_optimize(
            template,
            fragments,
            active_sites,
            chiralities=chiralities,
            output_dir=output_dir,
            conf_gen_options=conf_gen_options,
            take_unique=take_unique,
            num_structs=num_structs,
            calc=calc,
            evaluate_energy=evaluate_energy,
            energy_evaluator=energy_evaluator,
            preoptimize=preoptimize,
            optimizer_settings=optimizer_settings,
            smiles_hash_generator=smiles_hash_generator,
            parallelizer=parallelizer,
            batch_size=batch_size,
            verbose=verbose,
            max_products=max_products,
            callback=callback
        )
    else:
        if output_dir is None:
            output_dir = '.'
        for n,f in enumerate(glob.glob(f"{output_dir}/**/product.json", recursive=True)):
            product_data = utils.read_namedtuple(f)
            print(f"Submitting updates for {f}")
            callback(product_data, f)
            if max_products is not None and n >= max_products:
                break

def compress_pipeline_data(
        top_dir,
        patterns="**/pipeline_data.json",
        loader=None,
        recursive=True,
        output_mode=None,
        output_file=None
):
    if loader is None:
        if isinstance(patterns, str):
            patterns = [patterns]
        loader = os.path.splitext(patterns[0])[1].strip(".")
    if loader == 'json':
        tree = utils.construct_json_file_tree(
            top_dir,
            js_patterns=patterns,
            recursive=recursive
        )
    else:
        tree = utils.construct_namedtuple_file_tree(
            top_dir,
            patterns=patterns,
            recursive=recursive
        )
    tree = {
        a:{b:v['pipeline_data'] for b,v in v1.items()} for a,v1 in tree.items()
    }
    if output_file is not None:
        if output_mode is None:
            if isinstance(output_file, str):
                if os.path.splitext(top_dir)[1] == '.json':
                    output_mode = 'json'
                else:
                    output_mode = 'npz'
            else:
                output_mode = loader
        if output_mode == 'json':
            dev.write_json(output_file, tree)
        else:
            utils.write_tree(output_file, tree,
                             mode=output_mode,
                             precompression_function=(
                                 utils.prep_compressed_namedtuple_data
                                    if loader != 'json' else
                                 None
                             ))

    return tree

def read_compressed_pipeline_data(pipeline_file):
    return utils.read_tree(pipeline_file, decompression_function=utils.decompress_namedtuple_data)

# def submit_if_not_found(glob_pattern, target_file,
#                         overwrite=False,
#                         submission_function=run_python_script):
#     import sys, os
#     import subprocess
#     import glob
#     import shlex
#
#     target = sys.argv[1]
#     prods = glob.glob(f"{target}/*/*/optimized_forces.json")
#     overwrite = False
#     overwrite = (
#         (
#             True
#             if sys.argv[2].lower() == 'true' else
#             False
#             if sys.argv[2].lower() == 'false' else
#             overwrite
#         ) if len(sys.argv) > 2 else
#         overwrite
#     )
#     for pfile in prods:
#         pdir = os.path.dirname(pfile)
#         if overwrite or not os.path.isfile(os.path.join(pdir, 'force_modified_0.json')):
#             print(f"Submitting: {pfile}")
#             subprocess.call(
#                 shlex.split(f"sbatch --job-name=run_full_force_optimize run_python.sh '{pfile}' {overwrite}"))
#             # raise Exception(pfile)