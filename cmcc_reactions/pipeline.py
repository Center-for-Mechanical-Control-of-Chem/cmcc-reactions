from __future__ import annotations

import tempfile
import traceback as tb
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
from Psience.Molecools import Molecule

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
        'force_optimizer_settings',
        'predistortion_datasets',
        'random_force_coeffs'
    ],
    defaults=[None, None]
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
    _ti = None
    _tf = None

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
                random_force_coeffs='random_coeffs',
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
                'force_magnitudes': 'force_magnitude',
                'predistortion_datasets': 'predistorted_data'
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
                random_coeffs=data.random_force_coeffs,
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
                self._optimizer = fopt.ForceOptimizer.from_data(
                    self.optimized_forces,
                    reactant=self.reactant,
                    ts=self.transition_state,
                    reembed=False,
                )
            return self._optimizer

    def animate_fmrd_direction(self, fmrd_index, mass_weight=False, **etc):
       return self.optimizer.animate_normed(
           0,
           displacements=[self.fmrds[fmrd_index].force_vector],
           use_internals=len(self.fmrds[fmrd_index].force_vector) < len(self.product.atoms) * 3,
           mass_weight=mass_weight,
           **etc
       )

    def predicted_fmrd_distortion(self, fmrd_index, mass_weight=False, **etc):
       return self.optimizer.predicted_delta_from_forces(
           0,
           self.fmrds[fmrd_index].force_magnitude,
           units=("Hartrees", "BohrRadius"),
           displacements=[self.fmrds[fmrd_index].force_vector],
           use_internals=len(self.fmrds[fmrd_index].force_vector) < len(self.product.atoms) * 3,
           mass_weight=mass_weight,
           **etc
       )

    def trajectory_analyzer(self, **opts):
        if self.trajectory is None: return None
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
                     which='final',
                     force_modified='both',
                     **opts):
        traj = rda.DielsAlderReactionTrajectory.from_trajectory_data(self.trajectory, which=which)
        figure, x = traj.plot_profile(
            return_metrics=True,
            distance_metric=distance_metric,
            **opts
        )
        if force_modified and fmrd_index is None and fmrd is None and self.fmrds is not None:
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
        if force_modified and fmrd_index is None and fmrd is None and self.fmrds is not None:
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

    @property
    def initial_trajectory(self):
        if self._ti is None:
            self._ti = self.trajectory_analyzer(which='initial')
        return self._ti

    @property
    def refined_trajectory(self):
        if self._tf is None:
            self._tf = self.trajectory_analyzer(which='final')
        return self._tf


    @property
    def reactant(self):
        if self.refined_trajectory is not None:
            r = self.refined_trajectory.reactant
            if r.potential_derivatives is None and self.optimized_forces is not None:
                r.potential_derivatives = [0, np.asanyarray(self.optimized_forces.reactant_hessian)]
        else:
            r = None
        return r
    @property
    def transition_state(self):
        if self.refined_trajectory is not None:
            ts = self.refined_trajectory.transition_state
            if ts.potential_derivatives is None and self.optimized_forces is not None:
                ts.potential_derivatives = [0, np.asanyarray(self.optimized_forces.transition_state_hessian)]
        else:
            ts = None
        return ts
    @property
    def product_molecule(self):
        if self.refined_trajectory is not None:
            prod = self.refined_trajectory.product
        else:
            prod = Molecule(
                self.product.atoms,
                self.product.coords,
                bonds=self.product.bonds,
                energy_evaluator=self.product.evaluator
            )

        return prod

    def animate_reactant_distortion(self, fmrd_index, embed=True, embedding_indices=None, **opts):
        coords = [
            self.reactant.coords,
            self.fmrds[fmrd_index].force_modified_reactant_geom
        ]
        if embed:
            if nput.is_int(embedding_indices):
                embedding_indices = self.reactant.fragment_indices[embedding_indices]
            coords = self.reactant.embed_coords(coords, sel=embedding_indices)
        return self.reactant.plot(coords, **opts)

    def animate_ts_distortion(self, fmrd_index, embed=True, embedding_indices=None, **opts):
        coords = [
            self.transition_state.coords,
            self.fmrds[fmrd_index].force_modified_transition_state_geom
        ]
        if embed:
            if nput.is_int(embedding_indices):
                embedding_indices = self.transition_state.fragment_indices[embedding_indices]
            coords = self.transition_state.embed_coords(coords, sel=embedding_indices)
        return self.transition_state.plot(coords, **opts)

    def animate_ts_refinement(self, embed=True, embedding_indices=None, **opts):
        coords = [
            self.initial_trajectory.transition_state.coords,
            self.transition_state.coords
        ]
        if embed:
            if nput.is_int(embedding_indices):
                embedding_indices = self.transition_state.fragment_indices[embedding_indices]
            coords = self.transition_state.embed_coords(coords, sel=embedding_indices)
        return self.transition_state.plot(coords, **opts)



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

def prep_force_optimizer(trajectory,
                         internals='auto',
                         breakpoints=((0, 2), (1, 3)),
                         diene_bonds=((0, 4), (1, 5)),
                         fragment_indices='auto',
                         fix_breakpoint_atoms=True,
                         projection_internals='auto',
                         remove_fragment_transrot=True,
                         remove_local_transrot=True,
                         allow_mode_mixing=True,
                         which='final',
                         **opts):
    if not hasattr(trajectory, 'reactant'):
        trajectory = rda.DielsAlderReactionTrajectory.from_trajectory_data(trajectory, which=which)
    return fopt.ForceOptimizer.from_da_fragments(
        trajectory.reactant, trajectory.transition_state,
        internals=internals,
        breakpoints=breakpoints,
        diene_bonds=diene_bonds,
        fragment_indices=fragment_indices,
        fix_breakpoint_atoms=fix_breakpoint_atoms,
        projection_internals=projection_internals,
        remove_fragment_transrot=remove_fragment_transrot,
        remove_local_transrot=remove_local_transrot,
        allow_mode_mixing=allow_mode_mixing,
        **opts
    )

def run_force_optimization(trajectory,
                           internals='auto',
                           breakpoints=((0, 2), (1, 3)),
                           fragment_indices='auto',
                           fix_breakpoint_atoms=True,
                           projection_internals='auto',
                           remove_fragment_transrot=True,
                           remove_local_transrot=True,
                           allow_mode_mixing=True,
                           which='final',
                           **opts):
    opt = prep_force_optimizer(
        trajectory,
        internals=internals,
        breakpoints=breakpoints,
        fragment_indices=fragment_indices,
        fix_breakpoint_atoms=fix_breakpoint_atoms,
        projection_internals=projection_internals,
        remove_fragment_transrot=remove_fragment_transrot,
        remove_local_transrot=remove_local_transrot,
        allow_mode_mixing=allow_mode_mixing,
        which=which,
        **opts
    )
    opt.optimize()
    return opt

def run_fmrds(optimizer,
              nmodes=15,
              magnitude=(-200, -100, -50, 50, 100, 200),
              split_magnitudes=True,
              pool=None,
              **opts):
    nmodes = min(optimizer.force_coeffs.shape[0], nmodes)
    return optimizer.reoptimize_with_force(
        list(range(nmodes)),
        magnitude=magnitude,
        split_magnitudes=split_magnitudes,
        pool=pool,
        **opts
    )

def run_internal_fmrds(optimizer,
                       internal_selector='dihedrals',
                       magnitude=(-200, -100, -50, 50, 100, 200),
                       max_internals=10,
                       verbose=True,
                       pool=None,
                       memprof=None,
                       split_magnitudes=True,
                       **opts):
    if memprof is not None:
        print(f"Writing memory profile to {memprof}")
        import memray
        try:
            os.remove(memprof)
        except FileNotFoundError:
            ...
        with memray.Tracker(memprof) as tracker:
            return run_internal_fmrds(
                optimizer,
                internal_selector=internal_selector,
                magnitude=magnitude,
                max_internals=max_internals,
                verbose=verbose,
                pool=pool,
                memprof=None,
                split_magnitudes=split_magnitudes,
                **opts
            )
    else:
        return optimizer.reoptimize_internals_with_force(
            internal_selector,
            magnitude=magnitude,
            pool=pool,
            max_internals=max_internals,
            verbose=verbose,
            split_magnitudes=split_magnitudes,
            **opts
        )

def run_pressure_fmrds(optimizer,
                       pressure_model='xhcff',
                       magnitude=(200, 500, 1000, 5000, 10000),
                       verbose=True,
                       pool=None,
                       split_magnitudes=True,
                       **opts):
    return optimizer.reoptimize_with_pressure(
        magnitude=magnitude,
        pool=pool,
        pressure_model=pressure_model,
        verbose=verbose,
        split_magnitudes=split_magnitudes,
        **opts
    )

def run_random_fmrds(optimizer,
                     nmodes=15,
                     magnitude=(-200, -100, -50, 50, 100, 200),
                     verbose=True,
                     pool=None,
                     split_magnitudes=True,
                     **opts):
    nmodes = min(optimizer.random_coeffs.shape[0], nmodes)
    return optimizer.reoptimize_with_random_force(
        list(range(nmodes)),
        magnitude=magnitude,
        split_magnitudes=split_magnitudes,
        pool=pool,
        verbose=verbose,
        **opts
    )

default_step_ordering = {
    'products': 0,
    'trajectory': 1,
    'update_trajectory':1,
    'refine':1,
    'optimized_forces': 2,
    'fmrds': 3,
    'rigid-fmrds':3,
    'internals':3,
    'rigid-internals':3,
    'pressure':3,
    'rigid-pressure':3,
    'random':3,
    'rigid-random':3
}
def _check_step(force_steps, key, current):
    if dev.is_dict_like(force_steps):
        return force_steps.get(key, current is None)
    elif force_steps:
        return True
    elif current is None:
        return True
def run_optimization_pipeline(
        input_data: str | gen_prods.InitialProductData | OptimizedForceResults | OptimizedForcePipelineData,
        output_file=None,
        step_output_files=None,
        steps=None,
        force_steps=False,
        verbose=False,
        trajectory_optimization_settings=None,
        refined_trajectory_optimization_settings=None,
        update_trajectory_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        internal_force_modification_settings=None,
        pressure_force_modification_settings=None,
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
    elif isinstance(steps, str):
        steps = [steps]

    # steps = sorted(steps, key=lambda s: default_step_ordering[s])
    # if 'auto' in steps:
    #     i = steps.index('auto')
    #     prev = steps[:i]
    #     max_key = default_step_ordering[prev[0]]


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
        if (
                'trajectory' in steps
                and _check_step(force_steps, 'trajectory', input_data.trajectory)
        ):
            if trajectory_optimization_settings is None:
                trajectory_optimization_settings = {}
            trajectory_optimization_settings = global_options | trajectory_optimization_settings
            if verbose:
                print('running sampling')
            input_data.trajectory = run_initial_sampling(product, **trajectory_optimization_settings)

            if output_file is not None:
                input_data.save(output_file)

        if 'update_trajectory' in steps and _check_step(force_steps, 'update_trajectory', None):
            if update_trajectory_settings is None:
                update_trajectory_settings = {}
            # update_trajectory_settings = global_options | update_trajectory_settings
            if verbose:
                print('running updates')
            input_data.trajectory = run_update_trajectory(input_data.trajectory, **update_trajectory_settings)

            if output_file is not None:
                input_data.save(output_file)

        if 'refine' in steps and _check_step(force_steps, 'refine', None):
            if refined_trajectory_optimization_settings is None:
                refined_trajectory_optimization_settings = {}
            refined_trajectory_optimization_settings = global_options | refined_trajectory_optimization_settings
            if verbose:
                print('running refinement')
            input_data.trajectory = run_refined_trajectory(input_data.trajectory, **refined_trajectory_optimization_settings)

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        trajectory:gen_prods.ReoptimizedTrajectoryData = input_data.trajectory
        optimizer = None
        if (
                'optimized_forces' in steps
                and _check_step(force_steps, 'optimized_forces', input_data.optimizer)
        ):
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
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if (
                'fmrds' in steps
                and _check_step(force_steps, 'fmrds', input_data.fmrds)
        ):
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
                print(f"saving to {output_file}...")
                input_data.save(output_file)


        if 'rigid-fmrds' in steps and _check_step(force_steps, 'rigid-fmrds', None):
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running rigid fmrds')


            if force_modification_settings is None:
                force_modification_settings = {}
            force_modification_settings = global_options | force_modification_settings
            fmrds = run_fmrds(optimizer, rigid=True, **force_modification_settings)
            input_data.fmrds = [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if 'internals' in steps and _check_step(force_steps, 'internals', None):
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
            overwrite = internal_force_modification_settings.pop('overwrite', False)
            fmrds = run_internal_fmrds(optimizer, **internal_force_modification_settings)
            if input_data.fmrds is None or overwrite:
                input_data.fmrds = [f[2] for f in fmrds]
            else:
                input_data.fmrds = input_data.fmrds + [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if 'rigid-internals' in steps and _check_step(force_steps, 'rigid-internals', None):
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running rigid internal forces')

            if internal_force_modification_settings is None:
                internal_force_modification_settings = force_modification_settings
            if internal_force_modification_settings is None:
                internal_force_modification_settings = {}
            internal_force_modification_settings = global_options | internal_force_modification_settings
            overwrite = internal_force_modification_settings.pop('overwrite', False)
            fmrds = run_internal_fmrds(optimizer,
                                       rigid=True,
                                       **internal_force_modification_settings)
            if input_data.fmrds is None or overwrite:
                input_data.fmrds = [f[2] for f in fmrds]
            else:
                input_data.fmrds = input_data.fmrds + [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if 'pressure' in steps and _check_step(force_steps, 'pressure', None):
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running pressure')

            if pressure_force_modification_settings is None:
                pressure_force_modification_settings = force_modification_settings
            if pressure_force_modification_settings is None:
                pressure_force_modification_settings = {}
            pressure_force_modification_settings = global_options | pressure_force_modification_settings
            overwrite = pressure_force_modification_settings.pop('overwrite', False)
            fmrds = run_pressure_fmrds(optimizer, **pressure_force_modification_settings)
            if input_data.fmrds is None or overwrite:
                input_data.fmrds = [f[2] for f in fmrds]
            else:
                input_data.fmrds = input_data.fmrds + [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if 'rigid-pressure' in steps and _check_step(force_steps, 'rigid-pressure', None):
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running rigid pressure')

            if pressure_force_modification_settings is None:
                pressure_force_modification_settings = force_modification_settings
            if pressure_force_modification_settings is None:
                pressure_force_modification_settings = {}
            pressure_force_modification_settings = global_options | pressure_force_modification_settings
            overwrite = pressure_force_modification_settings.pop('overwrite', False)
            fmrds = run_pressure_fmrds(optimizer, rigid=True, **pressure_force_modification_settings)
            if input_data.fmrds is None or overwrite:
                input_data.fmrds = [f[2] for f in fmrds]
            else:
                input_data.fmrds = input_data.fmrds + [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if 'random' in steps and _check_step(force_steps, 'random', None):
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running random fmrds')


            if force_modification_settings is None:
                force_modification_settings = {}
            force_modification_settings = global_options | force_modification_settings
            fmrds = run_random_fmrds(optimizer, **force_modification_settings)
            input_data.fmrds = [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)

        if 'rigid-random' in steps and _check_step(force_steps, 'rigid-random', None):
            if optimizer is None:
                opt_force = input_data.optimized_forces
                if opt_force is None:
                    raise ValueError("optimized forces needed to get force modified reaction data")
                optimizer = fopt.ForceOptimizer.from_data(opt_force)
            if verbose:
                print('running rigid random fmrds')

            if force_modification_settings is None:
                force_modification_settings = {}
            force_modification_settings = global_options | force_modification_settings
            fmrds = run_random_fmrds(optimizer, rigid=True, **force_modification_settings)
            input_data.fmrds = [f[2] for f in fmrds]

            if output_file is not None:
                print(f"saving to {output_file}...")
                input_data.save(output_file)
    finally:
        if output_file is not None:
            print(f"saving to {output_file}...")
            input_data.save(output_file)
    if dev.str_is(step_output_files, 'auto'):
        step_output_files = {
            'pipeline_data_rigid.json':['rigid-fmrds'],
            'pipeline_data_internals.json':['internals'],
            'pipeline_data_internals_rigid.json':['rigid-internals'],
            'pipeline_data_random.json':['random'],
            'pipeline_data_random_rigid.json':['rigid-random'],
        }
    if step_output_files is not None:
        errors = []
        for of, steps in step_output_files.items():
            if not (
                    _check_step(force_steps, steps[-1], True if os.path.isfile(of) else None)
            ): continue
            try:
                run_optimization_pipeline(
                    input_data,
                    output_file=of,
                    steps=steps,
                    **(
                        global_options | dict(
                            verbose=verbose,
                            trajectory_optimization_settings=trajectory_optimization_settings,
                            refined_trajectory_optimization_settings=refined_trajectory_optimization_settings,
                            update_trajectory_settings=update_trajectory_settings,
                            optimized_force_settings=optimized_force_settings,
                            force_modification_settings=force_modification_settings,
                            internal_force_modification_settings=internal_force_modification_settings,
                            pressure_force_modification_settings=pressure_force_modification_settings,
                            max_iterations=max_iterations,
                            tol=tol,
                            energy_evaluator=energy_evaluator
                        )
                    )
                )
            except Exception as e:
                errors.append(e)
        if len(errors) > 0:
            for e in errors:
                tb.print_exception(e)
            raise errors[0]

    return input_data

DEBUG_NO_DELETE_SCRIPTS = False
def generator_callback(
        input_file=None,
        output_file="pipeline_data.json",
        steps=None,
        trajectory_optimization_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        max_iterations=500,
        tol=1e-8,
        nice=5,
        sbatch_kwargs=None,
        submit=True,
        presubmission_filter=None,
        presubmit_check_properties=None,
        energy_evaluator=None,
        verbose=True,
        **global_options
):
    if presubmission_filter is None and presubmit_check_properties is not None:
        if isinstance(presubmit_check_properties, str):
            presubmit_check_properties = [presubmit_check_properties]
        def presubmission_filter(product_data, input_file, out_file):
            if out_file is None or not os.path.isfile(out_file):
                return True
            else:
                data = utils.read_json(out_file)
                return any(
                    data.get(p) is None
                    for p in presubmit_check_properties
                )
    if sbatch_kwargs is None:
        sbatch_kwargs = {'mem':'15G', 'time':'8:00:00'}
    if 'nice' not in sbatch_kwargs:
        sbatch_kwargs['nice'] = nice
    def callback(product_data, product_file, input_file=input_file):
        targ_dir = os.path.dirname(product_file)
        product_file = os.path.basename(product_file)
        if input_file is None:
            input_file = product_file
        curdir = os.getcwd()
        try:
            os.chdir(targ_dir)
            out_file = output_file
            if presubmission_filter is not None and not presubmission_filter(product_data, input_file, out_file):
                print(f"Skipping `{out_file}` (already complete)")
                return
            if submit:
                script, _ = sbatch_python_job(
                    run_optimization_pipeline,
                    input_file,
                    out_file,
                    steps=steps,
                    trajectory_optimization_settings=trajectory_optimization_settings,
                    optimized_force_settings=optimized_force_settings,
                    force_modification_settings=force_modification_settings,
                    max_iterations=max_iterations,
                    tol=tol,
                    energy_evaluator=energy_evaluator,
                    verbose=verbose,
                    sbatch_kwargs=sbatch_kwargs, # disable stuff
                    post_processor='none',
                    **global_options
                )
                if verbose:
                    print(script.run(delete=not DEBUG_NO_DELETE_SCRIPTS))
                else:
                    script.run(delete=not DEBUG_NO_DELETE_SCRIPTS)
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
    return callback

def generate_from_product_library(
        template=None,
        fragments=None,
        active_sites=None,
        chiralities=None,
        output_dir=None,
        update_dir=None,
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
        input_file=None,
        max_products=None,
        steps=None,
        output_file="pipeline_data.json",
        step_output_files='auto',
        trajectory_optimization_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        max_iterations=500,
        tol=1e-8,
        sbatch_kwargs=None,
        submit=True,
        presubmission_filter=None,
        run_from_directory=False,
        smiles_cache=None,
        **global_options
):
    callback = generator_callback(
        input_file=input_file,
        output_file=output_file,
        steps=steps,
        trajectory_optimization_settings=trajectory_optimization_settings,
        optimized_force_settings=optimized_force_settings,
        force_modification_settings=force_modification_settings,
        max_iterations=max_iterations,
        tol=tol,
        sbatch_kwargs=sbatch_kwargs,
        submit=submit,
        presubmission_filter=presubmission_filter,
        energy_evaluator=energy_evaluator,
        step_output_files=step_output_files,
        verbose=verbose,
        **global_options
    )
    if run_from_directory:
        gen_prods.generate_products_and_optimize(
            template,
            fragments,
            active_sites,
            chiralities=chiralities,
            output_dir=output_dir,
            update_dir=update_dir,
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
            smiles_cache=smiles_cache,
            callback=None
        )
        if update_dir is None:
            update_dir = output_dir
        generate_from_directory(
            update_dir,
            energy_evaluator=energy_evaluator,
            verbose=verbose,
            input_file=input_file,
            max_products=max_products,
            steps=steps,
            output_file=output_file,
            trajectory_optimization_settings=trajectory_optimization_settings,
            optimized_force_settings=optimized_force_settings,
            force_modification_settings=force_modification_settings,
            max_iterations=max_iterations,
            tol=tol,
            sbatch_kwargs=sbatch_kwargs,
            submit=submit,
            callback=callback
        )
    else:
        gen_prods.generate_products_and_optimize(
            template,
            fragments,
            active_sites,
            chiralities=chiralities,
            output_dir=output_dir,
            update_dir=update_dir,
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
            smiles_cache=smiles_cache,
            callback=callback
        )

def generate_from_directory(
        output_dir,
        energy_evaluator='aimnet2',
        verbose=True,
        input_file=None,
        max_products=None,
        steps=None,
        output_file="pipeline_data.json",
        trajectory_optimization_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        max_iterations=500,
        tol=1e-8,
        sbatch_kwargs=None,
        submit=True,
        callback=None,
        **global_options
):
    if callback is None:
        callback = generator_callback(
            input_file=input_file,
            output_file=output_file,
            steps=steps,
            trajectory_optimization_settings=trajectory_optimization_settings,
            optimized_force_settings=optimized_force_settings,
            force_modification_settings=force_modification_settings,
            max_iterations=max_iterations,
            tol=tol,
            sbatch_kwargs=sbatch_kwargs,
            submit=submit,
            energy_evaluator=energy_evaluator,
            verbose=verbose,
            **global_options
        )
    if output_dir is None:
        output_dir = '.'
    if input_file is None:
        input_file = 'product.json'
    for n, f in enumerate(glob.glob(f"{output_dir}/**/{input_file}", recursive=True)):
        product_data = utils.read_namedtuple(f)
        print(f"Submitting updates for {f}")
        callback(product_data, f)
        if max_products is not None and n >= max_products:
            break

def generate_from_product_library_set(
        template_data:list[dict],
        output_dir=None,
        conf_gen_options=None,
        take_unique=True,
        num_structs=10,
        calc=None,
        evaluate_energy=True,
        energy_evaluator='aimnet2:aimnet2-nse',
        preoptimize=True,
        optimizer_settings=None,
        smiles_hash_generator='inchi',
        parallelizer=None,
        batch_size=50,
        verbose=True,
        input_file=None,
        max_products=None,
        steps=None,
        output_file="pipeline_data.json",
        trajectory_optimization_settings=None,
        optimized_force_settings=None,
        force_modification_settings=None,
        max_iterations=500,
        tol=1e-8,
        sbatch_kwargs=None,
        submit=True,
        run_from_directory=False,
        smiles_cache=None,
        **global_options
):
    all_temps = []
    all_frags = []
    all_active_sites = []
    all_chiralities = []

    if smiles_cache is None:
        smiles_cache = set()

    for d in template_data:
        all_temps.append(d['template'])
        all_frags.append(d['fragments'])
        all_active_sites.append(d['active_sites'])
        all_chiralities.append(d['chiralities'])

    for t,f,a,c in zip(all_temps, all_frags, all_active_sites, all_chiralities):
        generate_from_product_library(
            template=t,
            fragments=f,
            active_sites=a,
            chiralities=c,
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
            input_file=input_file,
            max_products=max_products,
            steps=steps,
            output_file=output_file,
            trajectory_optimization_settings=trajectory_optimization_settings,
            optimized_force_settings=optimized_force_settings,
            force_modification_settings=force_modification_settings,
            max_iterations=max_iterations,
            tol=tol,
            sbatch_kwargs=sbatch_kwargs,
            submit=submit,
            run_from_directory=run_from_directory,
            smiles_cache=smiles_cache,
            **global_options
        )

BASE_TEMPLATES = {
    'cyclopentadiene':'[C:3]1[C:1]([C:7]2)[C:5]=[C:6][C:2]2[C:4]1',
    'butadiene':'[C:3]1[C:1][C:5]=[C:6][C:2][C:4]1',
    '1-N-butadiene':'[C:3]1[C:1][C:5]=[N:6][C:2][C:4]1',
    '2-N-butadiene':'[C:3]1[C:1][C:5]=[C:6][N:2][C:4]1',
    '2-O-butadiene':'[C:3]1[C:1][C:5]=[C:6][O:2][C:4]1',
    'butadiene-N':'[C:3]1[C:1][C:5]=[C:6][C:2][N:4]1',
    'butadiene-O':'[C:3]1[C:1][C:5]=[C:6][C:2][O:4]1',
    'anthracene':'[C:3]4[C:1]3c1ccccc1[C:2]([C:4]4)c2ccccc23',
    'anthracene-side':'c12c(cc3ccccc3c1)[C:2]1[C:4][C:3][C:1]2C=C1',
    'dp-ibf':'[C:1]12(c3ccccc3)[C:3][C:4][C:2](c3ccccc3)(c3c1cccc3)O2',
    'dmfdc':'[C:1]12[C:3][C:4][C:2](C(C(OC)=O)=C1C(OC)=O)O2',
    'napthalene':'[C:1]12[C:3][C:4][C:2](C=C1)c1c2cccc1'
}

BASE_FRAGMENTS = {
    'sulfonyl':'[S:1](=O)=O',
    'sulfonyl-phenyl':'[S:1](=O)(c1ccccc1)=O',
    'sulfonyl-chex':'[S:1](=O)(C1CCCCC1)=O',
    'pF-phenyl':'[c:1]1ccc(F)cc1',
    'OMe':'[O:1]C',
    'CN':'[C:1]N',
    'acetamide':'[C:1]C(=O)N',
    'methyl':'[C:1]',
    'carboxyl':'[C:1]C(=O)O',
    'tBu':'[C:1]C(C)(C)(C)'
}

def _prep_pipeline_tree(tree, precompression_function):
    if isinstance(tree, tuple):
        if hasattr(tree, '_asdict'):
            tree = utils.namedtuple_dict(tree)
            if precompression_function is not None:
                tree = precompression_function(tree)
            return tree
        else:
            _, tree = tree
            if precompression_function is not None:
                tree = precompression_function(tree)
            return tree
    elif isinstance(tree, dict) and "_type" in tree:
        if precompression_function is not None:
            tree = precompression_function(tree)
        return tree
    else:
        return {
            k:_prep_pipeline_tree(v, precompression_function)
            for k, v in tree.items()
        }
def write_compressed_pipeline_data(output_file, tree, precompression_function=None, mode=None):
    if mode is None:
        if isinstance(output_file, str):
            if os.path.splitext(output_file)[1] == '.json':
                mode = 'json'
            else:
                mode = 'npz'
    if precompression_function is None and mode == 'npz':
        precompression_function = utils.prep_compressed_namedtuple_data
    tree = _prep_pipeline_tree(tree, precompression_function)
    if mode == 'json':
        dev.write_json(output_file, tree)
    else:
        utils.write_tree(output_file, tree, mode=mode)
def compress_pipeline_data(
        top_dir,
        patterns="**/pipeline_data.json",
        loader=None,
        recursive=True,
        js_loader=None,
        precompression_function=None,
        filter=None,
        ignore_bad=False,
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
            loader=js_loader,
            recursive=recursive,
            track_depths=True,
            filter=filter
        )
    else:
        tree = utils.construct_namedtuple_file_tree(
            top_dir,
            patterns=patterns,
            recursive=recursive,
            loader=js_loader,
            filter=filter,
            ignore_bad=ignore_bad,
            unwrap=True,
            track_depths=True
        )
    if output_file is not None:
        if output_mode is None:
            if isinstance(output_file, str):
                if os.path.splitext(output_file)[1] == '.json':
                    output_mode = 'json'
                else:
                    output_mode = 'npz'
            else:
                output_mode = loader
        write_compressed_pipeline_data(output_file, tree, precompression_function=precompression_function, mode=output_mode)

    return tree

def _unwrap_nts(pipeline_data, decompression_function, unwrap):
    if '_type' in pipeline_data:
        if decompression_function is not None:
            pipeline_data = decompression_function(pipeline_data)
        if unwrap:
            pipeline_data = utils.make_namedtuple(pipeline_data)
        return pipeline_data
    else:
        return {
            k:_unwrap_nts(v, decompression_function, unwrap)
            for k,v in pipeline_data.items()
        }
    # for k,v in pipeline_data.items():
    #     if '_type' in v:
def read_compressed_pipeline_data(pipeline_file, mode=None, decompression_function=None, unwrap=True, **opts):
    if mode is None:
        if isinstance(pipeline_file, str):
            if os.path.splitext(pipeline_file)[1] == '.json':
                mode = 'json'
            else:
                mode = 'npz'
    if decompression_function is None and mode == 'npz':
        decompression_function = utils.decompress_namedtuple_data
    base_data = utils.read_tree(pipeline_file, decompression_function=decompression_function, **opts)
    return _unwrap_nts(base_data, decompression_function, unwrap)
