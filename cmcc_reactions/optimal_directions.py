from __future__ import annotations
import typing


import gc
import collections
import os
import numpy as np
from scipy.optimize import minimize as scipy_opt
from McUtils.Data import UnitsData
from McUtils.Scaffolding import Logger
import McUtils.Devutils as dev
import McUtils.Numputils as nput
import McUtils.Iterators as itut
import McUtils.Coordinerds as coordops
from Psience.Molecools import Molecule
from Psience.Modes import MixtureModes
import McUtils.Plots as plt
import McUtils.Jupyter as interactive
from Psience.Reactions import Reaction

from . import utils
from . import trajectory_tools as trajt

__all__ = [
    "find_optimal_displacement_coordinate",
    "construct_force_dirs",
    "reaction_force_dirs",
    "compute_reaction_gamma"
]

def clip_f(f):
    f = np.clip(f, -1e15, 1e15)
    if np.abs(f) < 1e-16:
        s = np.sign(f)
        if s == 0: s = 1
        f = s * 1e-15
    return f

def gamma(hess_gs, hess_ts, d, d_ts=None, return_components=False):
    if d_ts is None:
        d_ts = d
    f_g = np.dot(np.dot(hess_gs, d), d)
    f_t = np.dot(np.dot(hess_ts, d_ts), d_ts)
    g = 1 / f_g - 1 / f_t
    if return_components:
        return g, f_g, f_t
    else:
        return g

def df_inv(hess, d):
    d1 = np.dot(hess, d)
    f = np.dot(d1, d)
    return -2 * d1 / f ** 2

def df_inv2(hess, d):
    d1 = np.dot(hess, d)
    f = np.dot(d1, d)
    d1 = 2 * d1
    d2 = 2 * hess
    return 2 * (d1[:, np.newaxis] * d1[np.newaxis, :]) / f ** 3 - d2 / f ** 2


def dgamma(hess_gs, hess_ts, d):
    return df_inv(hess_gs, d) - df_inv(hess_ts, d)


def dgamma2(hess_gs, hess_ts, d):
    return df_inv2(hess_gs, d) - df_inv2(hess_ts, d)


def get_guess_dir(f_proj_r, f_proj_ts):
    evals, L_ts = np.linalg.eigh(f_proj_ts)
    L_ts = L_ts[:, np.where(np.abs(evals) > 1e-8)[0]]
    g_base_r = np.diag(L_ts.T @ f_proj_r @ L_ts)
    g_base_ts = np.diag(L_ts.T @ f_proj_ts @ L_ts)
    guess_pos = np.argmax(1 / g_base_r - 1 / g_base_ts)
    return L_ts[:, guess_pos]

def scipy_optimize_forces(gs_hess, ts_hess, guess_dir, proj_dirs, *,
                          max_iterations,
                          logger=None,
                          method='nelder-mead',
                          **options):
    reduced_basis = nput.find_basis(nput.orthogonal_projection_matrix(proj_dirs))
    gs_hess = reduced_basis.T @ gs_hess @ reduced_basis
    ts_hess = reduced_basis.T @ ts_hess @ reduced_basis

    guess_dir = np.dot(guess_dir, reduced_basis)

    def fun(guess):
        guess = nput.vec_normalize(guess)
        return -np.array([gamma(gs_hess, ts_hess, guess)])

    def jac(guess):
        guess = nput.vec_normalize(guess)
        gg = dgamma(gs_hess, ts_hess, guess)
        g2 = np.dot(nput.orthogonal_projection_matrix(guess[:, np.newaxis]), gg)
        return -g2

    _, x, min = nput.scipy_minimize(
        guess_dir,
        function=fun,
        jacobian=jac,
        method=method,
        max_iterations=max_iterations,
        **options
    )

    return np.dot(nput.vec_normalize(x), reduced_basis.T), min

    # min = scipy_opt(fun, guess_dir, method=method, **opts)
    # opts = options | dict(options={'maxiter':max_iterations})
    # if method in {'cg', 'bfgs'}:
    #     opts['jac'] = jac
    #
    # if logger is not None:
    #     logger = Logger.lookup(logger)
    #     prev_re = [guess_dir]
    #     opts['callback'] = lambda intermediate_result, prev_re=prev_re: (
    #         logger.log_print(
    #             [
    #                 "Struct: {intermediate_result}",
    #                 "Step: {intermediate_step}"
    #             ],
    #             intermediate_result=intermediate_result,
    #             intermediate_step=intermediate_result - prev_re[-1]
    #         ),
    #         prev_re.append(intermediate_result)
    #     )
    #
    # min = scipy_opt(fun, guess_dir, method=method, **opts)
    # return np.dot(nput.vec_normalize(min.x), reduced_basis.T), min

def mcutils_optimize_forces(gs_hess, ts_hess, guess_dir, proj_dirs, *, max_iterations,
                            logger=None,
                            method='cg'):
    def fun(guess, mask):
        return -np.array([gamma(gs_hess, ts_hess, guess[0])])

    def jac(guess, mask):
        gg = dgamma(gs_hess, ts_hess, guess[0])
        # print("!", np.isnan(gg).any())
        return -gg[np.newaxis]

    def fhess(guess, mask):
        return -dgamma2(gs_hess, ts_hess, guess[0])[np.newaxis]

    if dev.str_is(method, 'cg'):
        method = nput.ConjugateGradientStepFinder(fun, jac,
                                                       damping_parameter=.9,
                                                       restart_interval=20
                                                       )
    elif dev.str_is(method, 'quasi-newton'):
        method = nput.QuasiNewtonStepFinder(fun, jac)

    force_dir, is_opt, error = nput.iterative_step_minimize(
        guess_dir,
        method,
        unitary=True,
        orthogonal_directions=proj_dirs,
        logger=logger,
        # generate_rotation=True,
        max_iterations=max_iterations
    )

    return force_dir, error


DEFAULT_MAX_ITERATIONS = 100
def find_optimal_displacement_coordinate(gs_hess, ts_hess, proj_dirs,
                                         guess_dir=None,
                                         max_iterations=None,
                                         optimizer='mcutils',
                                         perturbation=0,
                                         **opts
                                         ):
    if guess_dir is None:
        proj = nput.orthogonal_projection_matrix(proj_dirs)
        f_proj_r = proj @ gs_hess @ proj
        f_proj_ts = proj @ ts_hess @ proj
        guess_dir = get_guess_dir(f_proj_r, f_proj_ts)

    dx = perturbation * np.dot(nput.orthogonal_projection_matrix(proj_dirs), np.random.rand(*guess_dir.shape))
    guess_dir = nput.vec_normalize(guess_dir + dx)

    if max_iterations is None:
        max_iterations = DEFAULT_MAX_ITERATIONS
    if max_iterations < 0:
        return guess_dir, -1

    if dev.str_is(optimizer, 'scipy'):
        optimizer = scipy_optimize_forces
    elif dev.str_is(optimizer, 'mcutils'):
        optimizer = mcutils_optimize_forces

    return optimizer(gs_hess, ts_hess, guess_dir, proj_dirs,
                     max_iterations=max_iterations,
                     **opts)


def get_force_dirs(hess_gs, hess_ts, initial_dir, k, **opts):
    initial_dir = np.asanyarray(initial_dir)
    if initial_dir.ndim == 1:
        initial_dir = initial_dir[:, np.newaxis]
    proj_dirs = initial_dir
    errors = []
    for i in range(min(k, len(hess_gs)-proj_dirs.shape[-1])):
        force_dir, error = find_optimal_displacement_coordinate(
            hess_gs,
            hess_ts,
            proj_dirs,
            **opts
        )
        proj_dirs = np.concatenate([proj_dirs, force_dir[:, np.newaxis]], axis=1)
        errors.append(error)

    return proj_dirs, errors


def nm_hess(modes, L=None):
    if L is None:
        L = modes.matrix
    freqs2 = np.sign(modes.freqs) * modes.freqs ** 2
    return L @ np.diag(freqs2) @ L.T

def setup_mol(reactant_data):
    # reactant_data = np.load(args.reactant)
    nat = len(reactant_data['numbers'])
    reactant = Molecule(
        reactant_data['numbers'],
        reactant_data['coords'] * UnitsData.convert("Angstroms", "BohrRadius"),
        potential_derivatives=[
            0,
            reactant_data['hessian'].reshape(nat * 3, nat * 3) * UnitsData.convert("ElectronVolts", "Hartrees") / (
                    UnitsData.convert("Angstroms", "BohrRadius") ** 2
            )
        ],
        spin=1
    )
    return reactant


def construct_force_dirs(modes_gs, modes_ts,
                         *,
                         num_dirs,
                         idx_start,
                         mols=None,
                         internals=None,
                         max_iterations=None,
                         use_mode_space=True,
                         force_dir_generator=None,
                         **opts
                         ):

    if use_mode_space:
        if dev.str_is(use_mode_space, 'cartesian'):
            reactant, transition_state = mols
            hess_ts_nms = nput.tensor_reexpand([modes_ts.coords_by_modes], [0, transition_state.potential_derivatives[1]])[1]
            hess_gs_nms = nput.tensor_reexpand([modes_ts.coords_by_modes], [0, reactant.potential_derivatives[1]])[1]
        else:
            hess_ts_nms = nm_hess(modes_ts, L=np.eye(modes_ts.modes_by_coords.shape[-1]))
            hess_gs_nms = nm_hess(modes_gs, L=modes_ts.coords_by_modes @ modes_gs.modes_by_coords)
        proj_dir = np.eye(len(hess_ts_nms))[:, :idx_start]
    elif internals is not None:
        reactant, transition_state = mols
        exp_rs = reactant.modify(internals=internals).get_cartesians_by_internals(order=1)
        exp_ts = transition_state.modify(internals=internals).get_cartesians_by_internals(order=1)
        exp_ts_inv = transition_state.modify(internals=internals).get_internals_by_cartesians(order=1)

        hess_gs_nms = modes_gs.compute_hessian('coords')
        hess_ts_nms = modes_ts.compute_hessian('coords')

        hess_ts_nms = nput.tensor_reexpand(exp_ts, [0, hess_ts_nms])[1]
        hess_gs_nms = nput.tensor_reexpand(exp_rs, [0, hess_gs_nms])[1]

        proj_dir = (modes_ts.coords_by_modes[:idx_start, :] @ exp_ts_inv[0]).T
    else:
        hess_gs_nms = modes_gs.compute_hessian('coords')
        hess_ts_nms = modes_ts.compute_hessian('coords')

        proj_dir = modes_ts.coords_by_modes[:idx_start, :].T

    # hess_ts_nms = nm_hess(modes_ts, L=np.eye(modes_ts.matrix.shape[-1]))
    # hess_gs_nms = nm_hess(modes_gs, L=modes_ts.inverse @ modes_gs.matrix)

    num_dirs = min(num_dirs, len(hess_gs_nms) - proj_dir.shape[1])
    if force_dir_generator is None:
        force_dirs = get_force_dirs
    force_dirs, errors = force_dir_generator(hess_gs_nms, hess_ts_nms,
                                        proj_dir,
                                        # new_modes_ts.matrix[:, (0,)],
                                        num_dirs,
                                        max_iterations=max_iterations,
                                        **opts
                                        )
    if use_mode_space:
        cart_force_dirs = force_dirs.T @ modes_ts.coords_by_modes
    elif internals is not None:
        cart_force_dirs = force_dirs.T @ exp_ts[0]
    else:
        cart_force_dirs = force_dirs.T
    selected_force_dirs = cart_force_dirs[idx_start:]

    return selected_force_dirs, force_dirs

def write_gsm_constraint(nat, selected_force_dirs,
                         file_pattern="force_{i}.txt",
                         force_units='PicoNewtons',
                         energy_units='ElectronVolts',
                         distance_units='Angstroms',
                         ):
    if force_units is not None:
        factor = UnitsData.convert(force_units, 'AtomicUnitOfForce') * (
                UnitsData.convert("Hartrees", energy_units)
                / UnitsData.convert("BohrRadius", distance_units)
        )
    else:
        factor = 1

    files = []
    for i in range(len(selected_force_dirs)):
        filename = file_pattern.format(i=i)
        with open(filename, 'w') as f:
            tmp = selected_force_dirs[i].reshape(nat, 3)
            for j in range(len(tmp)):
                f.write(
                    "{0:.7f} {1:.7f} {2:.7f}\n".format(
                        tmp[j][0] * factor,
                        tmp[j][1] * factor,
                        tmp[j][2] * factor
                    )
                )
            files.append(filename)
    return files

def prep_optimization_modes(reactant, transition_state,
                            fragment_indices=None,
                            extra_localization=None,
                            remove_fragment_transrot=True,
                            remove_local_transrot=True,
                            allow_mode_mixing=True,
                            internals=None,
                            project_zero_gmatrix_modes=True):
    new_modes_ts = transition_state.get_normal_modes()
    new_modes_gs = reactant.get_normal_modes()

    if fragment_indices is not None:
        if isinstance(fragment_indices, int):
            fragment_indices = reactant.fragment_indices[fragment_indices]
        new_modes_gs = new_modes_gs.localize(atoms=fragment_indices,
                                             allow_mode_mixing=allow_mode_mixing,
                                             project_zero_gmatrix_modes=project_zero_gmatrix_modes)
        new_modes_ts = new_modes_ts.localize(atoms=fragment_indices,
                                             allow_mode_mixing=allow_mode_mixing,
                                             project_zero_gmatrix_modes=project_zero_gmatrix_modes)

        if remove_local_transrot:
            _, dx = nput.transrot_expansion(
                reactant.coords,
                *fragment_indices,
                masses=reactant.atomic_masses
            )
            proj = nput.frame_displacement_projector(dx.T, reactant.atomic_masses, mass_weighted=False)
            new_modes_gs = new_modes_gs.localize(projections=[proj], allow_mode_mixing=allow_mode_mixing)

            _, dx = nput.transrot_expansion(
                transition_state.coords,
                *fragment_indices,
                masses=transition_state.atomic_masses
            )
            proj = nput.frame_displacement_projector(dx.T, transition_state.atomic_masses, mass_weighted=True)
            new_modes_ts = new_modes_ts.localize(projections=[proj], allow_mode_mixing=allow_mode_mixing)

    if remove_fragment_transrot:
        _, dx = nput.orientation_expansion(
            reactant.coords,
            *reactant.fragment_indices,
            masses=reactant.atomic_masses
        )
        proj = nput.frame_displacement_projector(dx.T, reactant.atomic_masses, mass_weighted=False)
        new_modes_gs = new_modes_gs.localize(projections=[proj], allow_mode_mixing=allow_mode_mixing)

        _, dx = nput.orientation_expansion(
            transition_state.coords,
            *reactant.fragment_indices,
            masses=transition_state.atomic_masses
        )
        proj = nput.frame_displacement_projector(dx.T, transition_state.atomic_masses, mass_weighted=True)
        new_modes_ts = new_modes_ts.localize(projections=[proj], allow_mode_mixing=allow_mode_mixing)

    if internals is not None:
        if len(internals) == 0:
            raise ValueError("no internals supplied")
        if isinstance(internals, dict):
            internals = internals['specs']
        elif all(len(internals) == 4 for internals in internals):
            internals = coordops.extract_zmatrix_internals(internals)
        new_modes_gs = new_modes_gs.localize(internals=internals, allow_mode_mixing=allow_mode_mixing)
        new_modes_ts = new_modes_ts.localize(internals=internals, allow_mode_mixing=allow_mode_mixing)

    if extra_localization is not None:
        extra_localization = extra_localization | dict(allow_mode_mixing=allow_mode_mixing)
        new_modes_gs = new_modes_gs.localize(**extra_localization)
        new_modes_ts = new_modes_ts.localize(**extra_localization)

    return new_modes_gs, new_modes_ts

LOW_FREQUENCY_MODE_CUTOFF = 0.00045
def reaction_force_dirs(reactant, transition_state,
                        *,
                        num_dirs,
                        prepped_modes=None,
                        fragment_indices=None,
                        low_frequency_cutoff=None,  # 100 cm-1
                        return_modes=True,
                        extra_localization=None,
                        remove_fragment_transrot=True,
                        remove_local_transrot=True,
                        allow_mode_mixing=True,
                        internals=None,
                        project_internals=True,
                        **opts
                        ):

    if prepped_modes is None:
        prepped_modes = prep_optimization_modes(reactant, transition_state,
                                                fragment_indices=fragment_indices,
                                                extra_localization=extra_localization,
                                                remove_fragment_transrot=remove_fragment_transrot,
                                                remove_local_transrot=remove_local_transrot,
                                                internals=internals if project_internals else None,
                                                allow_mode_mixing=allow_mode_mixing)
    new_modes_gs, new_modes_ts = prepped_modes

    if low_frequency_cutoff is None:
        low_frequency_cutoff = LOW_FREQUENCY_MODE_CUTOFF
    if low_frequency_cutoff > 0:
        ts_freqs = new_modes_ts.freqs
        idx_start = int(np.where(ts_freqs >= low_frequency_cutoff)[0][0])
    else:
        idx_start = 1

    dirs = construct_force_dirs(new_modes_gs, new_modes_ts,
                                num_dirs=num_dirs,
                                idx_start=idx_start,
                                mols=(reactant, transition_state),
                                internals=internals,
                                **opts)
    if return_modes:
        return dirs, (new_modes_gs, new_modes_ts)
    else:
        return dirs

def get_random_displacement_coordinate(gs_hess, ts_hess, proj_dirs, *, rng=None):
    """
    Like `find_optimal_displacement_coordinate`, but instead of optimizing
    gamma, draws a single random direction orthogonal to `proj_dirs`.
    """
    if not hasattr(rng, 'normal'):
        rng = np.random.default_rng(rng)
    proj = nput.orthogonal_projection_matrix(proj_dirs)
    raw = rng.normal(size=proj_dirs.shape[0])
    guess_dir = nput.vec_normalize(proj @ raw)
    error = gamma(gs_hess, ts_hess, guess_dir)
    return guess_dir, error


def get_random_force_dirs(hess_gs, hess_ts, initial_dir, k, rng=None):
    """
    Like `get_force_dirs`, but each successive direction is a random draw
    orthogonal to everything already selected (rather than an optimized one).
    """
    initial_dir = np.asanyarray(initial_dir)
    if initial_dir.ndim == 1:
        initial_dir = initial_dir[:, np.newaxis]
    proj_dirs = initial_dir
    errors = []
    if not hasattr(rng, 'normal'):
        rng = np.random.default_rng(rng)
    for i in range(min(k, len(hess_gs) - proj_dirs.shape[-1])):
        force_dir, error = get_random_displacement_coordinate(
            hess_gs, hess_ts, proj_dirs, rng=rng
        )
        proj_dirs = np.concatenate([proj_dirs, force_dir[:, np.newaxis]], axis=1)
        errors.append(error)

    return proj_dirs, errors


def construct_random_force_dirs(modes_gs, modes_ts,
                                *,
                                num_dirs,
                                idx_start,
                                mols=None,
                                internals=None,
                                use_mode_space=True,
                                **opts
                                ):
    return construct_force_dirs(
        modes_gs, modes_ts,
        num_dirs=num_dirs,
        idx_start=idx_start,
        mols=mols,
        internals=internals,
        use_mode_space=use_mode_space,
        force_dir_generator=get_random_force_dirs,
        **opts
    )


def random_force_dirs(reactant, transition_state,
                      *,
                      num_dirs,
                      prepped_modes=None,
                      fragment_indices=None,
                      low_frequency_cutoff=None,
                      return_modes=True,
                      extra_localization=None,
                      remove_fragment_transrot=True,
                      remove_local_transrot=True,
                      allow_mode_mixing=True,
                      internals=None,
                      project_internals=True,
                      **opts
                      ):
    return reaction_force_dirs(
        reactant, transition_state,
        num_dirs=num_dirs,
        prepped_modes=prepped_modes,
        fragment_indices=fragment_indices,
        low_frequency_cutoff=low_frequency_cutoff,
        return_modes=return_modes,
        extra_localization=extra_localization,
        remove_fragment_transrot=remove_fragment_transrot,
        remove_local_transrot=remove_local_transrot,
        allow_mode_mixing=allow_mode_mixing,
        internals=internals,
        project_internals=project_internals,
        force_dir_constructor=construct_random_force_dirs,
        **opts
    )

def get_target_displacement_coordinate(gs_hess, ts_hess, proj_dirs, dir):
    """
    Like `find_optimal_displacement_coordinate`, but instead of optimizing
    gamma, draws a single random direction orthogonal to `proj_dirs`.
    """
    proj = nput.orthogonal_projection_matrix(proj_dirs)
    raw = np.asanyarray(dir)
    guess_dir = nput.vec_normalize(proj @ raw)
    error = gamma(gs_hess, ts_hess, guess_dir)
    return guess_dir, error


def get_target_force_dirs(hess_gs, hess_ts, initial_dir, k, *, target_dirs):
    """
    Like `get_force_dirs`, but each successive direction is a random draw
    orthogonal to everything already selected (rather than an optimized one).
    """
    initial_dir = np.asanyarray(initial_dir)
    if initial_dir.ndim == 1:
        initial_dir = initial_dir[:, np.newaxis]
    proj_dirs = initial_dir
    errors = []
    for dir in target_dirs:
        force_dir, error = get_target_displacement_coordinate(
            hess_gs, hess_ts, proj_dirs, dir
        )
        proj_dirs = np.concatenate([proj_dirs, force_dir[:, np.newaxis]], axis=1)
        errors.append(error)

    return proj_dirs, errors


def construct_target_force_dirs(modes_gs, modes_ts,
                                *,
                                target_dirs,
                                idx_start,
                                num_dirs=None,
                                mols=None,
                                internals=None,
                                use_mode_space=True,
                                **opts
                                ):
    return construct_force_dirs(
        modes_gs, modes_ts,
        num_dirs=num_dirs,
        idx_start=idx_start,
        mols=mols,
        internals=internals,
        use_mode_space=use_mode_space,
        target_dirs=len(target_dirs),
        force_dir_generator=get_target_force_dirs,
        **opts
    )


def target_force_dirs(reactant, transition_state,
                      *,
                      target_dirs,
                      prepped_modes=None,
                      fragment_indices=None,
                      low_frequency_cutoff=None,
                      return_modes=True,
                      extra_localization=None,
                      remove_fragment_transrot=True,
                      remove_local_transrot=True,
                      allow_mode_mixing=True,
                      internals=None,
                      project_internals=True,
                      **opts
                      ):
    return reaction_force_dirs(
        reactant, transition_state,
        num_dirs=len(target_dirs),
        target_dirs=target_dirs,
        prepped_modes=prepped_modes,
        fragment_indices=fragment_indices,
        low_frequency_cutoff=low_frequency_cutoff,
        return_modes=return_modes,
        extra_localization=extra_localization,
        remove_fragment_transrot=remove_fragment_transrot,
        remove_local_transrot=remove_local_transrot,
        allow_mode_mixing=allow_mode_mixing,
        internals=internals,
        project_internals=project_internals,
        force_dir_constructor=construct_target_force_dirs,
        **opts
    )

def prep_gamma_hessians(
        reactant, transition_state, direction_gs,
        direction_ts=None,
        use_mode_space=True,
        modes=None
):
    direction_gs = np.asanyarray(direction_gs)
    if direction_ts is None:
        direction_ts = direction_gs
    else:
        direction_ts = np.asanyarray(direction_ts)
    if modes is None:
        modes = (None, None)
    new_modes_gs, new_modes_ts = modes
    if new_modes_ts is None:
        new_modes_ts = transition_state.get_normal_modes()
    if new_modes_gs is None:
        new_modes_gs = reactant.get_normal_modes()

    if use_mode_space:
        if dev.str_is(use_mode_space, 'cartesian'):
            f_ts = nput.tensor_reexpand([new_modes_ts.coords_by_modes], [0, transition_state.potential_derivatives[1]])[1]
            f_gs = nput.tensor_reexpand([new_modes_ts.coords_by_modes], [0, reactant.potential_derivatives[1]])[1]
            direction_gs = np.dot(direction_gs, new_modes_ts.modes_by_coords)
            direction_ts = np.dot(direction_ts, new_modes_ts.modes_by_coords)
        else:
            f_ts = nm_hess(new_modes_ts, L=np.eye(new_modes_ts.modes_by_coords.shape[-1]))
            f_gs = nm_hess(new_modes_gs, L=new_modes_ts.coords_by_modes @ new_modes_gs.modes_by_coords)
            direction_gs = np.dot(direction_gs, new_modes_ts.modes_by_coords)
            direction_ts = np.dot(direction_ts, new_modes_ts.modes_by_coords)
    else:
        f_gs = new_modes_gs.compute_hessian('coords')
        f_ts = new_modes_ts.compute_hessian('coords')

    return (direction_gs, direction_ts), (f_gs, f_ts)

def compute_reaction_gamma(reactant, transition_state, direction_gs,
                           direction_ts=None,
                           use_mode_space=True,
                           modes=None,
                           return_components=False
                           ):
    (direction_gs, direction_ts), (f_gs, f_ts) = prep_gamma_hessians(
        reactant, transition_state,
        direction_gs, direction_ts, use_mode_space, modes
    )

    if direction_gs.ndim > 1:
        return np.array([
            gamma(f_gs, f_ts, d, dt, return_components=return_components)
            for d,dt in zip(direction_gs, direction_ts)
        ])
    else:
        return gamma(
            f_gs,
            f_ts,
            direction_gs,
            direction_ts,
            return_components=return_components
        )

def reorder_force_dirs(rs_solv, ts_solv, dirs, direction_ts=None,
                       use_mode_space=True,
                       modes=None,
                       return_ordering=False
                       ):
    gamma_list = compute_reaction_gamma(rs_solv, ts_solv, dirs, direction_ts=direction_ts,
                                        use_mode_space=use_mode_space,
                                        modes=modes)
    ord_g = np.argsort(-gamma_list)
    if direction_ts is not None:
        dirs = (dirs[ord_g,], direction_ts[ord_g,])
    else:
        dirs = dirs[ord_g,]
    if return_ordering:
        return gamma_list[ord_g,], dirs, ord_g
    else:
        return gamma_list[ord_g,], dirs

def mass_weighted_normalize_displacements(mol, expansion=None, inverse=None,
                                          mass_weight=True, orthogonalize=False, mode='forward',
                                          return_norms=False):
    if mode == 'forward':
        if expansion is None:
            expansion = mol.get_cartesians_by_internals(1)[0]
        else:
            expansion = np.asanyarray(expansion) #TODO: why?
            if expansion.ndim == 3:
                expansion = expansion[0]
        if mass_weight:
            b = expansion @ mol.get_gmatrix(power=-1/2, use_internals=False)
        else:
            b = expansion
        if orthogonalize:
            b, r = np.linalg.qr(b.T)
            b = b.T
            if inverse is not None:
                pinv = np.linalg.inv(b @ mol.get_gmatrix(power=1/2, use_internals=False) @ inverse)
                inverse = inverse @ pinv
        else:
            b, norms = nput.vec_normalize(b, axis=1, return_norms=True)
            if inverse is not None:
                inverse = inverse @ np.diag(norms)
        if mass_weight:
            b = b @ mol.get_gmatrix(power=1/2, use_internals=False)

        if inverse is not None:
            if return_norms:
                return b, inverse, norms
            else:
                return b, inverse
        else:
            if return_norms:
                return b, norms
            else:
                return b
    elif mode == 'inverse':
        if inverse is not None:
            if expansion is None:
                expansion = mol.get_cartesians_by_internals(1)[0]
            else:
                expansion = np.asanyarray(expansion)  # TODO: why?
                if expansion.ndim == 3:
                    expansion = expansion[0]
            if mass_weight:
                b = mol.get_gmatrix(power=1/2, use_internals=False) @ inverse
            else:
                b = inverse
            if orthogonalize:
                b, r = np.linalg.qr(b)
                pinv = np.linalg.inv(expansion @ mol.get_gmatrix(power=-1/2, use_internals=False) @ b)
                expansion = pinv @ expansion
            else:
                b, norms = nput.vec_normalize(b, axis=0, return_norms=True)
                expansion = np.diag(norms) @ expansion
            if mass_weight:
                b = mol.get_gmatrix(power=-1/2, use_internals=False) @ b

            if return_norms:
                return b, expansion, norms
            else:
                return expansion, b
        else:
            if expansion is None:
                expansion = mol.get_internals_by_cartesians(1)[0]
            else:
                expansion = np.asanyarray(expansion)  # TODO: why?
                if expansion.ndim == 3:
                    expansion = expansion[0]
            if mass_weight:
                b = mol.get_gmatrix(power=1/2, use_internals=False) @ expansion
            else:
                b = expansion
            if orthogonalize:
                b, r = np.linalg.qr(b)
            else:
                b, norms = nput.vec_normalize(b, axis=0, return_norms=True)
            if mass_weight:
                b = mol.get_gmatrix(power=-1/2, use_internals=False) @ b
            if return_norms:
                return b, norms
            else:
                return b
    else:
        raise ValueError(mode)

def mass_weighted_displacement_inverse(mol, expansion, use_pinv=False):
    gi12 = mol.get_gmatrix(power=-1/2, use_internals=False)
    b = expansion @ gi12
    if use_pinv:
        bT = np.linalg.pinv(b)
    else:
        bT = b.T
    return gi12 @ bT

def gamma_to_force_conversions(energy_units="Kilocalories/Mole", force_units="PicoJoules/Meters"):
    return (
            UnitsData.convert("Hartrees", energy_units)
            / UnitsData.convert("Hartrees/BohrRadius", force_units) ** 2
    )

def anharmonic_response_ratios(b, f, a, d):
    return [
        (2*a*b) / (3*f),
        (2*((5/24)*d - a**2/(8*f))*b**2) / f,
        (2*(-a**3/(8*f**2) + a*d/(12*f))*b**3) / f,
        (2*(-a**4/(48*f**3) - a**2*d/(48*f**2) + d**2/(24*f))*b**4) / f,
        (a*d**2*b**5) / (18*f**3),
        (2*(a**4*d/(384*f**4) + a**2*d**2/(72*f**3) + d**3/(144*f**2))*b**6) / f,
        (2*(a**3*d**2/(288*f**4) + a*d**3/(162*f**3))*b**7) / f,
        (2*(a**2*d**3/(576*f**4) + d**4/(1296*f**3))*b**8) / f,
        (a*d**4*b**9) / (1296*f**5),
        (d**5*b**10) / (15552*f**5),
    ]
def anharmonic_response_from_force(x, f, a, d, force_units='Picojoules/Meters'):
    e_units, d_units = force_units.split("/")
    b = x * (
        UnitsData.convert(e_units, "Hartrees")
        / UnitsData.convert(d_units, "BohrRadius")
    ) / f
    return b, anharmonic_response_ratios(b, f, -a, d)

def lj_repulsion(dists, rad, epsilon=1, exponent=12):
    return epsilon * (rad / dists)**exponent
def clipped_sphere_repulsion(dists, rad, epsilon=1, attenuation=.1, exponent=6, clip=True):
    mask = dists < (rad * (1 + attenuation))
    pot = epsilon * ((rad * (1 + attenuation)) / dists)**exponent
    if clip:
        pot = np.clip(pot, 0, 1)
    return pot * mask
def interior_repulsion(dists, rad, epsilon=1, attenuation=.1, exponent=6):
    pot = epsilon / (1 + np.exp(-(rad - dists) / attenuation)**(exponent/6))
    return pot
def pointwise_steric_potential(centers, radii, points_groups, pairwise_term=lj_repulsion,
                               atom_groups=None,
                               center_exponent=6,
                               pointwise_exponent=None,
                               point_repulsion_radius=.1,
                               return_breakdowns=False):
    total_repulsion = 0 if not return_breakdowns else [np.zeros(len(pg), dtype='float') for pg in points_groups]
    for i,g in enumerate(points_groups):
        if atom_groups is None:
            subgroup = [i]
        else:
            subgroup = next((a for a in atom_groups if i in a), [i])
        g = np.asanyarray(g)
        rem = np.setdiff1d(np.arange(len(centers)), subgroup)
        other_centers = np.array([centers[j] for j in rem])
        other_radii = np.array([radii[j] for j in rem])
        if center_exponent is not None:
            pairwise_dists = np.linalg.norm(other_centers[:, np.newaxis, :] - g[np.newaxis, :, :], axis=-1)
            terms = pairwise_term(pairwise_dists, other_radii[:, np.newaxis], exponent=center_exponent)
            if return_breakdowns:
                total_repulsion[i] += np.sum(terms, axis=0)
            else:
                total_repulsion += np.sum(terms)
        if pointwise_exponent is not None:
            other_points = np.concatenate([points_groups[j] for j in rem], axis=0)
            pairwise_dists = np.linalg.norm(other_points[:, np.newaxis, :] - g[np.newaxis, :, :], axis=-1)
            terms = pairwise_term(pairwise_dists, point_repulsion_radius, exponent=pointwise_exponent)
            if return_breakdowns:
                total_repulsion[i] += np.sum(terms, axis=0)
            else:
                total_repulsion += np.sum(terms)
    return total_repulsion
def molecule_steric_potential(mol,
                              density=10,
                              molecule_distortion_function=None,
                              point_transformation_function=None,
                              prune=True,
                              separate_fragments=False,
                              atom_groups=None,
                              pairwise_term=interior_repulsion,
                              center_exponent=6,
                              pointwise_exponent=None,
                              preserve_clipping=False,
                              return_breakdowns=False):
    surf = mol.get_surface()
    if separate_fragments:
        frags = mol.fragments
        inds = mol.fragment_indices
        pts = [None] * len(mol.atoms)
        for m,x in zip(frags, inds):
            subsurf = m.get_surface()
            subpts = subsurf.generate_points(density=density, preserve_origins=True, prune=prune)
            for i,p in zip(x, subpts):
                pts[i] = p
    else:
        pts = surf.generate_points(density=density, preserve_origins=True, prune=prune)
    resummation = (not return_breakdowns) and preserve_clipping
    if preserve_clipping:
        masks = [surf.get_exterior_points(p, surf.centers, surf.radii) for p in pts]
        return_breakdowns = True
    base_val = pointwise_steric_potential(surf.centers, surf.radii, pts, pairwise_term=pairwise_term,
                                          atom_groups=atom_groups,
                                          center_exponent=center_exponent,
                                          pointwise_exponent=pointwise_exponent,
                                          return_breakdowns=return_breakdowns)
    if point_transformation_function is None and molecule_distortion_function is not None:
        def point_transformation_function(mol, pts):
            new_mol_geoms = molecule_distortion_function(mol)
            smol = new_mol_geoms.shape == 2
            if smol: new_mol_geoms = [new_mol_geoms]
            new_pts = []
            for nmg in new_mol_geoms:
                distortion = nmg - mol.coords
                new_pts.append([
                    p + d[np.newaxis, :]
                    for p,d in zip(pts, distortion)
                ])
            if smol:
                new_pts = new_pts[0]
                new_mol_geoms = new_mol_geoms[0]
            return new_mol_geoms, new_pts
    if point_transformation_function is not None:
        new_centers, new_pts = point_transformation_function(mol, pts)
        if isinstance(new_pts[0], np.ndarray) and new_pts[0].shape == pts[0].shape:
            if return_breakdowns:
                subpot = pointwise_steric_potential(new_centers, surf.radii, new_pts, pairwise_term=pairwise_term,
                                                    atom_groups=atom_groups,
                                                    center_exponent=center_exponent,
                                                    pointwise_exponent=pointwise_exponent,
                                                    return_breakdowns=return_breakdowns)
                diffs = [
                    s - b
                    for s,b in zip(subpot, base_val)
                ]
                if not resummation:
                    pts = [pts, new_pts]
                    base_val = [[np.zeros_like(d) for d in diffs], diffs]
                else:
                    base_val = diffs
            else:
                base_val = pointwise_steric_potential(new_centers, surf.radii, new_pts,
                                                      atom_groups=atom_groups,
                                                      center_exponent=center_exponent,
                                                      pointwise_exponent=pointwise_exponent,
                                                      pairwise_term=pairwise_term) - base_val
        else:
            if return_breakdowns:
                res = []
                for c,p in zip(new_centers, new_pts):
                    subpot = pointwise_steric_potential(c, surf.radii, p, pairwise_term=pairwise_term,
                                                        atom_groups=atom_groups,
                                                        center_exponent=center_exponent,
                                                        pointwise_exponent=pointwise_exponent,
                                                        return_breakdowns=return_breakdowns)
                    diffs = [
                        s - b
                        for s, b in zip(subpot, base_val)
                    ]
                    res.append(diffs)
                if not resummation:
                    base_val = [[np.zeros_like(d) for d in res[0]]] + res
                    pts = [pts] + list(new_pts)
                else:
                    base_val = res
            else:
                base_val = [
                    pointwise_steric_potential(c, surf.radii, p,
                                               atom_groups=atom_groups,
                                               center_exponent=center_exponent,
                                               pointwise_exponent=pointwise_exponent,
                                               pairwise_term=pairwise_term) - base_val
                    for c,p in zip(new_centers, new_pts)
                ]
        if preserve_clipping:
            base_val = [[mask * v for mask,v in zip(masks, bv)] for bv in base_val]
        if resummation:
            base_val = [np.sum(np.concatenate(v)) for v in base_val]
            return_breakdowns = False
        if return_breakdowns:
            return pts, base_val
        else:
            return base_val
    else:
        if return_breakdowns:
            return np.concatenate(pts, axis=0), np.concatenate(base_val, axis=0)
        else:
            return base_val

def molecule_volume_change(mol,
                           geoms=None,
                           molecule_distortion_function=None,
                           use_mol_ref=True,
                           **volume_opts):
    if geoms is None:
        geoms = molecule_distortion_function(mol)

    vols = np.array([
        mol.modify(coords=g).get_surface().volume(**volume_opts)
        for g in geoms
    ])

    if use_mol_ref:
        return vols - mol.get_surface().volume(**volume_opts)
    else:
        return vols

OptimizedForceData = collections.namedtuple(
    'OptimizedForceData',
    [
        'atoms',
        'reactant_geom',
        'transition_state_geom',
        'reactant_hessian',
        'transition_state_hessian',
        'energy_evaluator',
        'force_coeffs',
        'internals',
        'use_mode_space',
        'optimizer_settings',
        'random_coeffs'
    ],
    defaults=[None, None]
)
utils.register_namedtuple(OptimizedForceData)

ForceModifiedReactionData = collections.namedtuple(
    'ForceModifiedReactionData',
    [
        'atoms',
        'reactant_geom',
        'reactant_energy',
        'transition_state_geom',
        'transition_state_energy',
        'force_modified_reactant_geom',
        'force_modified_reactant_energy',
        'force_modified_transition_state_geom',
        'force_modified_transition_state_energy',
        'force_vector',
        'force_magnitude',
        'force_units',
        'mass_weight',
        'energy_evaluator',
        'internals',
        'predistorted_data',
        'optimizer_settings'
    ],
    defaults=[None, None]
)
utils.register_namedtuple(ForceModifiedReactionData)

class ForceOptimizer:
    default_options = {
        'num_dirs':15,
        'optimizer':'scipy',
        'method':'cg',
        'max_iterations':100
    }
    def __init__(self, reactant_mol, ts_mol,
                 optimal_forces=None,
                 reorder=True,
                 inverse_forces=None,
                 use_mode_space=True,
                 reembed=True,
                 prepped_modes=None,
                 internals=None,
                 projection_internals=None,
                 reactant_hessian=None,
                 transition_state_hessian=None,
                 precompute_modes=True,
                 force_coeffs=None,
                 random_forces=None,
                 random_coeffs=None,
                 **determination_opts):
        if reactant_hessian is not None:
            reactant_mol = reactant_mol.modify(potential_derivatives=[0, reactant_hessian])
        if transition_state_hessian is not None:
            ts_mol = ts_mol.modify(potential_derivatives=[0, transition_state_hessian])
        if reembed:
            if precompute_modes:
                reactant_mol.get_normal_modes() # precompute modes
                ts_mol.get_normal_modes() # precompute modes
            ts_mol = ts_mol.get_embedded_molecule()
            reactant_mol = reactant_mol.get_embedded_molecule(ref=ts_mol)
        self.rs:Molecule = reactant_mol
        self.ts:Molecule = ts_mol
        self._prepped_modes = prepped_modes
        self.reorder = reorder
        self.use_mode_space = use_mode_space
        if projection_internals is None:
            projection_internals = internals
        # if dev.str_is(projection_internals, 'auto'):
        #     projection_internals = self.prep_projection_internals(internals)
        if projection_internals is not None:
            determination_opts = determination_opts | dict(internals=projection_internals)
        self.opts = self.default_options | determination_opts
        self.internals = internals
        if optimal_forces is None and force_coeffs is not None:
            optimal_forces = {'force_coeffs':force_coeffs}
        self._optimal_forces = optimal_forces
        if random_forces is None and random_coeffs is not None:
            random_forces = {'force_coeffs':random_forces}
        self._random_forces = random_forces
        self._inverse = inverse_forces
        self._internal_mols = None
        self._internal_modes = None
        self._internal_dirs = None
        self._pure_internal_displacement_matrix = None

    def get_forces_from_coeffs(self, coeffs):
        coeffs = np.asanyarray(coeffs)
        if self.use_mode_space:
            cart_force_dirs = coeffs.T @ self.prepped_modes[1].coords_by_modes
        elif self.internals is not None:
            cart_force_dirs = coeffs.T @ self.ts.get_cartesians_by_internals(order=1)[0]
        else:
            cart_force_dirs = coeffs

        gammas = compute_reaction_gamma(self.rs, self.ts, cart_force_dirs, use_mode_space=self.use_mode_space)
        return (gammas, cart_force_dirs), self.prepped_modes, coeffs

    def to_data(self) -> OptimizedForceData:
        fcs = self.force_coeffs
        rcs = self.random_coeffs if self._random_forces is not None else None
        return OptimizedForceData(
            atoms=self.rs.atoms,
            reactant_geom=self.rs.coords,
            transition_state_geom=self.ts.coords,
            reactant_hessian=self.rs.potential_derivatives[1],
            transition_state_hessian=self.ts.potential_derivatives[1],
            energy_evaluator=self.rs.energy_evaluator,
            force_coeffs=fcs,
            random_coeffs=rcs,
            internals=self.internals,
            use_mode_space=self.use_mode_space,
            optimizer_settings=self.opts
        )
    def save(self, output_dir, info_file='optimized_forces.json'):
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

    @classmethod
    def from_data(cls, force_data:ForceOptimizer|OptimizedForceData, reactant=None, ts=None, **opts):
        if isinstance(force_data, ForceOptimizer):
            return force_data
        else:
            energy_evaluator = force_data.energy_evaluator
            opts = opts | force_data.optimizer_settings
            ee = opts.pop('energy_evaluator', None)
            if energy_evaluator is None:
                energy_evaluator = ee
            if reactant is None:
                reactant = Molecule(
                    force_data.atoms,
                    force_data.reactant_geom,
                    potential_derivatives=[0, np.asanyarray(force_data.reactant_hessian)],
                    energy_evaluator=energy_evaluator,
                    spin=1
                )
            elif reactant.potential_derivatives is None:
                reactant.potential_derivatives = [0, np.asanyarray(force_data.reactant_hessian)]

            if ts is None:
                ts = Molecule(
                    force_data.atoms,
                    force_data.transition_state_geom,
                    potential_derivatives=[0, np.asanyarray(force_data.transition_state_hessian)],
                    energy_evaluator=energy_evaluator,
                    spin=1
                )
            elif ts.potential_derivatives is None:
                ts.potential_derivatives = [0, np.asanyarray(force_data.transition_state_hessian)]

            if 'internals' in opts:
                opts['projection_internals'] = opts.pop('internals')
            return cls(
                reactant, ts,
                force_coeffs=force_data.force_coeffs,
                internals=force_data.internals,
                use_mode_space=force_data.use_mode_space,
                **opts
            )
    @classmethod
    def from_file(cls, force_data):
        return cls.from_data(utils.read_namedtuple(force_data))

    @classmethod
    def from_trajectory(cls, trajectory, **opts):
        if isinstance(trajectory, str):
            trajectory = trajt.DielsAlderReactionTrajectory.from_file(trajectory)
        elif not isinstance(trajectory, trajt.DielsAlderReactionTrajectory):
            trajectory = trajt.DielsAlderReactionTrajectory.from_trajectory_data(trajectory)

        return cls(trajectory.reactant, trajectory.transition_state, **opts)

    @classmethod
    def from_displacements(cls,
                           reactant_mol, ts_mol, dirs_gs, dirs_ts=None,
                           reorder=False,
                           modes=None,
                           orthogonalize=False,
                           inverse=None,
                           dirs_gs_inv=None,
                           dirs_ts_inv=None,
                           use_mode_space=True,
                           orthogonalization_mode='forward'
                           ):
        if modes is None:
            modes = (None, None)
        rs_modes, ts_modes = modes
        if rs_modes is None:
            if reactant_mol.potential_derivatives is None:
                reactant_mol.potential_derivatives = reactant_mol.calculate_energy(order=2)[1:]
            rs_modes = reactant_mol.get_normal_modes(use_internals=False)
        if ts_modes is None:
            if ts_mol.potential_derivatives is None:
                ts_mol.potential_derivatives = ts_mol.calculate_energy(order=2)[1:]
            ts_modes = ts_mol.get_normal_modes(use_internals=False)

        if len(dirs_gs) == 2 and nput.is_numeric_array_like(dirs_gs[0], 2):
            dirs_gs, dirs_ts = dirs_gs
        if inverse is not None:
            if dirs_ts is not None:
                dirs_gs_inv, dirs_ts_inv = inverse
            else:
                dirs_gs_inv = inverse

        dirs_gs = mass_weighted_normalize_displacements(reactant_mol, dirs_gs,
                                                        inverse=dirs_gs_inv,
                                                        orthogonalize=orthogonalize,
                                                        mode=orthogonalization_mode)
        if dirs_gs_inv is not None:
            dirs_gs, dirs_gs_inv = dirs_gs

        if dirs_ts is not None:
            dirs_ts = mass_weighted_normalize_displacements(ts_mol, dirs_ts,
                                                            inverse=dirs_ts_inv,
                                                            orthogonalize=orthogonalize,
                                                            mode=orthogonalization_mode)
            if dirs_ts_inv is not None:
                dirs_ts, dirs_ts_inv = dirs_ts

        if reorder:
            gammas, dirs, ord = reorder_force_dirs(reactant_mol, ts_mol, dirs_gs, dirs_ts,
                                                   modes=(rs_modes, ts_modes),
                                                   use_mode_space=use_mode_space,
                                                   return_ordering=True)
            if dirs_ts_inv is not None:
                inverse = (dirs_gs_inv[:, ord,], dirs_ts_inv[:, ord,])
            elif dirs_gs_inv is not None:
                inverse = dirs_gs_inv[:, ord]
            else:
                inverse = None

        else:
            gammas = compute_reaction_gamma(reactant_mol, ts_mol, dirs_gs, dirs_ts,
                                            use_mode_space=use_mode_space,
                                            modes=(rs_modes, ts_modes))
            if dirs_ts is not None:
                dirs = (dirs_gs, dirs_ts)
            else:
                dirs = dirs_gs

            if dirs_ts_inv is not None:
                inverse = (dirs_gs_inv, dirs_ts_inv)
            elif dirs_gs_inv is not None:
                inverse = dirs_gs_inv
            else:
                inverse = None

        return cls(reactant_mol, ts_mol,
                   optimal_forces=((gammas, dirs), (rs_modes, ts_modes)),
                   inverse_forces=inverse,
                   use_mode_space=use_mode_space
                   )

    @classmethod
    def from_mol_displacements(cls, reactant_mol, ts_mol, internals=None, **opts):
        if internals is not None:
            ts_mol = ts_mol.modify(internals=internals)
            reactant_mol = reactant_mol.modify(internals=ts_mol.internals)
        rs_disp_inv = reactant_mol.get_internals_by_cartesians(order=1)[0]
        ts_disp_inv = ts_mol.get_internals_by_cartesians(order=1)[0]
        rs_disp = reactant_mol.get_cartesians_by_internals(order=1)[0]
        ts_disp = ts_mol.get_cartesians_by_internals(order=1)[0]
        return cls.from_displacements(reactant_mol, ts_mol, rs_disp, ts_disp,
                                      dirs_gs_inv=rs_disp_inv,
                                      dirs_ts_inv=ts_disp_inv,
                                      **opts)

    @classmethod
    def from_guesses(cls,
                     rs,
                     ts,
                     reactant_optimizer='pysis',
                     reactant_optimizer_method='rfo',
                     ts_optimizer='pysis',
                     ts_optimizer_method='ts',
                     max_iterations=200,
                     logger=None,
                     optimizer_settings=None,
                     precompute_modes=False,
                     **etc
                     ):
        if optimizer_settings is None:
            optimizer_settings = {}
        rs = rs.optimize(mode=reactant_optimizer, method=reactant_optimizer_method, max_iterations=max_iterations,
                         logger=logger, **optimizer_settings)
        ts = ts.optimize(mode=ts_optimizer, method=ts_optimizer_method, max_iterations=max_iterations,
                         logger=logger, **optimizer_settings)

        return cls(rs, ts, precompute_modes=precompute_modes, **etc)

    @classmethod
    def from_internals(cls, reactant_mol, ts_mol, internal_spec, active_atoms=None, fixed_atoms=None,
                       remove_translation_rotation=True,
                       **opts):
        if fixed_atoms is None and active_atoms is not None:
            fixed_atoms = np.setdiff1d(np.arange(len(reactant_mol.coords)), active_atoms)

        # rs_dist.fragment_indices[1][(4, 5, 6, 7),]
        inv_gs, disp_gs = nput.internal_coordinate_tensors(
            reactant_mol.coords,
            internal_spec,
            fixed_atoms=fixed_atoms,
            masses=reactant_mol.atomic_masses,
            return_inverse=True,
            remove_inverse_translation_rotation=remove_translation_rotation,
            order=1
        )
        inv_ts, disp_ts = nput.internal_coordinate_tensors(
            ts_mol.coords,
            internal_spec,
            fixed_atoms=fixed_atoms,
            masses=ts_mol.atomic_masses,
            return_inverse=True,
            remove_inverse_translation_rotation=remove_translation_rotation,
            order=1
        )

        return cls.from_displacements(reactant_mol, ts_mol, disp_gs[0], disp_ts[0],
                                      dirs_gs_inv=inv_gs[1],
                                      dirs_ts_inv=inv_ts[1],
                                      **opts)

    @classmethod
    def from_modes(cls, reactant_mol, ts_mol,
                   modes=None,
                   fragment_indices=None,
                   ts_only=False,
                   active_atoms=None,
                   fixed_atoms=None,
                   **opts):
        if modes is None:
            modes = (modes, modes)
        gs_modes, ts_modes = modes
        if gs_modes is None:
            if ts_only:
                gs_modes = ts_mol.get_normal_modes(use_internals=False)
            else:
                gs_modes = reactant_mol.get_normal_modes(use_internals=False)
        if ts_modes is None:
            ts_modes = ts_mol.get_normal_modes(use_internals=False)

        if fragment_indices is not None:
            if isinstance(fragment_indices, int):
                fragment_indices = reactant_mol.fragment_indices[fragment_indices]
            gs_modes = gs_modes.localize(atoms=fragment_indices, allow_mode_mixing=True)
            ts_modes = ts_modes.localize(atoms=fragment_indices, allow_mode_mixing=True)
        elif active_atoms is not None or fixed_atoms is not None:
            if fixed_atoms is not None:
                active_atoms = np.setdiff1d(np.arange(len(reactant_mol.coords)), fixed_atoms)
            gs_modes = gs_modes.localize(atoms=active_atoms, allow_mode_mixing=True)
            ts_modes = ts_modes.localize(atoms=active_atoms, allow_mode_mixing=True)
        gs_modes = gs_modes.remove_mass_weighting()
        ts_modes = ts_modes.remove_mass_weighting()
        return cls.from_displacements(
            reactant_mol, ts_mol, gs_modes.coords_by_modes, dirs_ts=ts_modes.coords_by_modes,
            dirs_gs_inv=gs_modes.modes_by_coords,
            dirs_ts_inv=ts_modes.modes_by_coords,
            **opts
        )

    @classmethod
    def construct(cls, rs, ts, *,
                  specs=None,
                  mol_internals=None,
                  displacements=None,
                  modes=None,
                  active_atoms=None,
                  fixed_atoms=None,
                  fragment_ref=None,
                  **opts):
        if active_atoms is not None or fixed_atoms is not None and fragment_ref is not None:
            frag = rs.fragment_indices[fragment_ref]
            if active_atoms is not None:
                active_atoms = frag[active_atoms,]
            if fixed_atoms is not None:
                fixed_atoms = frag[fixed_atoms,]
        if specs is not None:
            m = cls.from_internals(rs, ts, specs,
                                   active_atoms=active_atoms, fixed_atoms=fixed_atoms,
                                   **opts)
        elif mol_internals is not None:
            if active_atoms is not None:
                raise ValueError("`mol_internals` can't be paired with `active_atoms`")
            if active_atoms is not None:
                raise ValueError("`mol_internals` can't be paired with `active_atoms`")
            m = cls.from_mol_displacements(rs, ts, mol_internals, **opts)
        elif modes is not None:
            if dev.str_is(modes, 'auto'):
                modes = None
            m = cls.from_modes(rs, ts, modes=modes,
                               active_atoms=active_atoms, fixed_atoms=fixed_atoms,
                               **opts)
        elif displacements is not None:
            if active_atoms is not None:
                raise ValueError("`displacements` can't be paired with `active_atoms`")
            if active_atoms is not None:
                raise ValueError("`displacements` can't be paired with `active_atoms`")
            m = cls.from_displacements(rs, ts, displacements, **opts)
        else:
            if fixed_atoms is not None:
                active_atoms = np.setdiff1d(np.arange(len(rs.atoms)), fixed_atoms)
            if active_atoms is not None:
                fi = opts.get('fragment_indices')
                if fi is None:
                    opts['fragment_indices'] = active_atoms
                else:
                    raise ValueError("got both `fragment_indices` and `active_atoms` for optimizer")
            m = cls(rs, ts, specs, **opts)

        return m

    @classmethod
    def from_da_fragments(cls,
                          reactant,
                          transition_state,
                          internals='auto',
                          breakpoints=((0, 2), (1, 3)),
                          diene_bonds=((0, 4), (1, 5)),
                          fragment_indices='auto',
                          fix_breakpoint_atoms=True,
                          projection_internals='auto',
                          remove_fragment_transrot=True,
                          remove_local_transrot=True,
                          allow_mode_mixing=True,
                          **opts):
        ref = reactant
        if dev.str_is(fragment_indices, 'auto'):
            # focus only on the more substituted fragment
            # look for non-hydrogens at the key positions
            ats = ref.atoms
            bond_set = {frozenset(b[:2]) for b in ref.bonds}
            flat_breaks = [i for b in breakpoints for i in b]
            flat_dats = [i for b in diene_bonds for i in b]
            diene_atoms = {i for i in flat_breaks if i in flat_dats}
            diene_subs = len([
                b for b in (bond_set - {frozenset(db) for db in diene_bonds})
                if (
                        len(b & diene_atoms) == 1
                        and ats[next((i for i in b if i not in diene_atoms))] != "H"
                )
            ])
            dio_atoms = {i for i in flat_breaks if not i in flat_dats}
            dio_subs = len([
                b for b in (bond_set - {frozenset(dio_atoms)})
                if (
                        len(b & dio_atoms) == 1
                        and ats[next((i for i in b if i not in dio_atoms))] != "H"
                )
            ])
            if (diene_subs > dio_subs) or (dio_subs == 0):
                fragment_indices = 0
            else:
                fragment_indices = 1

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

        return cls(reactant, transition_state,
                  fragment_indices=fragment_indices,
                  remove_fragment_transrot=remove_fragment_transrot,
                  remove_local_transrot=remove_local_transrot,
                  allow_mode_mixing=allow_mode_mixing,
                  projection_internals=projection_internals,
                  internals=internals,
                  **opts
                  )

    @property
    def prepped_modes(self):
        if self._prepped_modes is None:
            opts = {
                k:v for k,v in self.opts.items()
                if k in ['fragment_indices',
                         'extra_localization',
                         'remove_fragment_transrot',
                         'remove_local_transrot',
                         'internals',
                         'project_internals',
                         'allow_mode_mixing']
            }
            project_internals = opts.pop('project_internals', None)
            if project_internals is False:
                opts.pop('internals', None)
            self._prepped_modes = prep_optimization_modes(
                self.rs, self.ts,
                **opts
            )
        return self._prepped_modes

    @property
    def internal_mols(self):
        if self._internal_mols is None:
            if self.internals is None:
                raise ValueError("`internals` can't be None")
            self._internal_mols = [
                self.rs.modify(internals=self.internals),
                self.ts.modify(internals=self.internals)
            ]
        return self._internal_mols
    @property
    def internal_modes(self):
        if self._internal_modes is None:
            m_r, m_t = self.prepped_modes
            r, t = self.internal_mols
            exp_r = r.get_internals_by_cartesians(order=1, strip_embedding=True)
            exp_t = t.get_internals_by_cartesians(order=1, strip_embedding=True)
            coeffs_r = m_r.coords_by_modes @ exp_r[0]
            coeffs_t = m_t.coords_by_modes @ exp_t[0]
            self._internal_modes = [
                coeffs_r,
                coeffs_t
            ]
        return self._internal_modes
    @property
    def internal_dirs(self):
        if self._internal_dirs is None:
            self._internal_dirs = (
                    self.force_dirs
                    @ self.internal_mols[1].get_internals_by_cartesians(1, strip_embedding=True)[0]
            )
        return self._internal_dirs
    @property
    def pure_internal_displacement_matrix(self):
        if self._pure_internal_displacement_matrix is None:
            if self.ts.potential_derivatives is None:
                self.ts.potential_derivatives = self.ts.calculate_energy(order=2)[1:]
                grad = self.ts.potential_derivatives[0]
            elif nput.is_zero(self.ts.potential_derivatives[0]):
                grad = self.ts.calculate_energy(order=1)[1]
            else:
                grad = self.ts.potential_derivatives[0]
            exp = self.internal_mols[1].get_cartesians_by_internals(1, strip_embedding=True)[0]
            signs = np.sign(exp @ grad[:, np.newaxis])[:, 0]
            self._pure_internal_displacement_matrix = np.diag(signs)
        return self._pure_internal_displacement_matrix

    @property
    def rs_modes(self):
        return self.prepped_modes[0]
    @property
    def ts_modes(self):
        return self.prepped_modes[1]

    def optimize(self):
        if self.rs.potential_derivatives is None:
            self.rs.potential_derivatives = self.rs.calculate_energy(order=2)[1:]
        if self.ts.potential_derivatives is None:
            self.ts.potential_derivatives = self.ts.calculate_energy(order=2)[1:]

        (dirs, coeffs), modes = reaction_force_dirs(self.rs, self.ts,
                                                    prepped_modes=self.prepped_modes,
                                                    use_mode_space=self.use_mode_space, **self.opts)
        if self.reorder:
            gammas, dirs, ord = reorder_force_dirs(self.rs, self.ts, dirs,
                                                   use_mode_space=self.use_mode_space,
                                                   return_ordering=True)
            coeffs = coeffs[:, ord]
        else:
            gammas = compute_reaction_gamma(self.rs, self.ts, dirs, use_mode_space=self.use_mode_space)

        return (gammas, dirs), modes, coeffs

    @property
    def optimal_force_data(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        elif isinstance(self._optimal_forces, dict):
            self._optimal_forces = self.get_forces_from_coeffs(self._optimal_forces['force_coeffs'])
        return self._optimal_forces
    @property
    def random_force_data(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.random_optimize()
        elif isinstance(self._random_forces, dict):
            self._random_forces = self.get_forces_from_coeffs(self._random_forces['force_coeffs'])
        return self._random_forces
    @property
    def gammas(self):
        return self.optimal_force_data[0][0]
    @property
    def force_dirs(self):
        return self.optimal_force_data[0][1]
    @property
    def force_dirs_inverse(self):
        if self._inverse is None:
            fds = self.optimal_force_data[0][1]
            if isinstance(fds, np.ndarray):
                self._inverse = mass_weighted_displacement_inverse(self.ts, fds)
            else:
                self._inverse = (
                    mass_weighted_displacement_inverse(self.rs, fds[0]),
                    mass_weighted_displacement_inverse(self.ts, fds[1])
                )
        return self._inverse
    @property
    def force_coeffs(self):
        return self.optimal_force_data[2]

    @property
    def random_gammas(self):
        return self.random_force_data[0][0]
    @property
    def random_dirs(self):
        return self.random_force_data[0][1]
    @property
    def random_dirs_inverse(self):
        if self._inverse is None:
            fds = self.random_force_data[0][1]
            if isinstance(fds, np.ndarray):
                self._inverse = mass_weighted_displacement_inverse(self.ts, fds)
            else:
                self._inverse = (
                    mass_weighted_displacement_inverse(self.rs, fds[0]),
                    mass_weighted_displacement_inverse(self.ts, fds[1])
                )
        return self._inverse
    @property
    def random_coeffs(self):
        return self.random_force_data[2]

    @property
    def mass_weighted_force_dirs(self):
        fds = self.force_dirs
        gi12 = self.ts.get_gmatrix(power=-1/2, use_internals=False)
        if isinstance(fds, np.ndarray):
            return fds @ gi12
        else:
            return (
                    fds[0] @ gi12,
                    fds[1] @ gi12
            )
    @property
    def mass_weighted_force_dirs_inverse(self):
        fds = self.force_dirs_inverse
        g12 = self.ts.get_gmatrix(power=1/2, use_internals=False)
        if isinstance(fds, np.ndarray):
            return g12 @ fds
        else:
            return (
                g12 @ fds[0],
                g12 @ fds[1]
            )

    @property
    def kcal_nN(self):
        return UnitsData.convert("Hartrees", "Kilocalories/Mole") * (
            UnitsData.convert(("NanoJoules", "InverseMeters"), ("Hartrees", "InverseBohrRadius"))
        ) ** 2

    def random_optimize(self):
        """
        Same as `optimize`, but draws a random force in the projected
        (mode-space / internals) subspace instead of numerically
        optimizing gamma from a Hessian-based initial guess.
        """
        if self.rs.potential_derivatives is None:
            self.rs.potential_derivatives = self.rs.calculate_energy(order=2)[1:]
        if self.ts.potential_derivatives is None:
            self.ts.potential_derivatives = self.ts.calculate_energy(order=2)[1:]

        opts = {
            k: v for k, v in self.opts.items()
            if k not in ('optimizer', 'method', 'max_iterations')
        }

        (dirs, coeffs), modes = random_force_dirs(self.rs, self.ts,
                                                  prepped_modes=self.prepped_modes,
                                                  use_mode_space=self.use_mode_space,
                                                  **opts)
        if self.reorder:
            gammas, dirs, ord = reorder_force_dirs(self.rs, self.ts, dirs,
                                                   use_mode_space=self.use_mode_space,
                                                   return_ordering=True)
            coeffs = coeffs[:, ord]
        else:
            gammas = compute_reaction_gamma(self.rs, self.ts, dirs, use_mode_space=self.use_mode_space)

        return (gammas, dirs), modes, coeffs

    _debug_show_force_vectors = False
    def mode_force_function(self, mode, magnitude=1,
                            use_internals=False,
                            mass_weight=True,
                            displacements=None,
                            reembed_displacements=None,
                            remove_transrot=True,
                            remove_orientation=True
                            ):
        displacements = self.get_displacement_dirs(mass_weight=mass_weight,
                                                   use_internals=use_internals,
                                                   displacements=displacements)
        if not callable(displacements):
            if reembed_displacements is None:
                reembed_displacements = True
            def get_direction(ref, coords):
                if self._debug_show_force_vectors:
                    self.rs.plot(
                        coords.reshape(-1, 3),
                        mode_vectors=nput.vec_normalize(displacements[mode]) * 15
                    ).show()
                return displacements[mode] * magnitude
        else:
            if reembed_displacements is None:
                reembed_displacements = False
            def get_direction(ref, coords):
                d = displacements(ref, coords)
                if self._debug_show_force_vectors:
                    self.rs.plot(
                        coords.reshape(-1, 3),
                        mode_vectors=nput.vec_normalize(d[mode]) * 15
                    ).show()
                return d[mode] * magnitude

        def force_modification(coords, base_grad):
            d = get_direction(self.ts, coords)
            if use_internals:
                coords = np.asanyarray(coords).reshape((-1, 3))
                dx = self.internal_mols[0].get_cartesians_by_internals(1, strip_embedding=True, coords=coords)[0]
                rot = np.dot(d, dx).reshape(base_grad.shape)
            elif reembed_displacements:
                coords = coords.reshape((-1,) + self.ts.coords.shape)
                emb = self.ts.get_embedding_data(coords)
                ## test embedding conventions
                # tf = (
                #         emb.coord_data.axes
                #         @ np.moveaxis(emb.rotations, -1, -2)
                #         @ np.moveaxis(emb.reference_data.axes, -1, -2)
                # )
                tf = (
                        np.moveaxis(emb.coord_data.axes, -1, -2)
                        @ emb.rotations
                        @ emb.reference_data.axes
                )
                rot = d.reshape(self.ts.coords.shape)[np.newaxis] @ tf
            else:
                rot = d.reshape(self.ts.coords.shape)[np.newaxis]

            if remove_transrot:
                ## try removing tranrot in
                rot = rot.reshape(rot.shape[:-2] + (1, -1))
                proj = nput.translation_rotation_projector(coords,
                                                           self.ts.masses,
                                                           mass_weighted=False,
                                                           orthonormal=False)
                # rot = rot @ proj
                rot = rot @ np.moveaxis(proj, -1, -2)

            if remove_orientation:
                rot = rot.reshape(rot.shape[:-2] + (1, -1))
                _, dx = nput.orientation_expansion(
                    coords,
                    *self.rs.fragment_indices,
                    masses=self.ts.masses
                )
                if dev.str_in(remove_orientation, ['translation', 'translations'], ignore_case=True):
                    dx = dx[..., (0, 1, 2), :]
                elif dev.str_in(remove_orientation, ['rotation', 'rotations'], ignore_case=True):
                    dx = dx[..., (3, 4, 5), :]
                proj = nput.frame_displacement_projector(np.moveaxis(dx, -1, -2), self.ts.masses, mass_weighted=False)
                # rot = rot @ proj
                rot = rot @ np.moveaxis(proj, -1, -2)

            rot = rot.reshape(base_grad.shape)
            return -rot
        return force_modification, get_direction(self.ts, self.ts.coords)

    def reoptimize_with_force(self,
                              mode,
                              magnitude=50,
                              units='PicoJoules/Meters',
                              use_internals=False,
                              mass_weight=False,
                              optimizer_mode='pysis',
                              optimizer_method='rfo',
                              profile_generator='pys-ts',
                              climb=True,
                              ts_opt_settings=None,
                              max_iterations=500,
                              max_displacement=.05,
                              predistort=False,
                              ts_initial_coords=None,
                              rs_initial_coords=None,
                              reoptimize_reactants=True,
                              reoptimize_ts=True,
                              initial_ts_step=1,
                              num_ts_steps=3,
                              initial_reactants_step=1,
                              modify_forces=True,
                              apply_constraints=True,
                              reembed_displacements=None,
                              remove_transrot=True,
                              remove_orientation=None,
                              output_dir=None,
                              info_file='force_modified_{mode}_{mag}.json',
                              displacements=None,
                              verbose=False,
                              run_gc=True,
                              logger=False,
                              rigid=False,
                              force_scan_steps=None,
                              split_magnitudes=False,
                              rigid_max_recursion=5,
                              rigid_scan_steps=10,
                              rigid_scan_options=None,
                              flatten_res=True,
                              # displacement_generator=None,
                              **opts):

        if logger is not None and logger is not False:
            logger = Logger.lookup(logger)

        smol_mode = nput.is_int(mode)
        smol_force = nput.is_numeric(magnitude)
        if smol_mode:
            modes = [mode]
        else:
            modes = mode

        if smol_force:
            mags = [magnitude]
        else:
            mags = magnitude

        if not rigid and split_magnitudes:
            magnitude = np.asanyarray(magnitude)
            neg_mag = np.where(magnitude < 0)
            pos_mag = np.where(magnitude >= 0)
            neg_ord = np.argsort(-magnitude[neg_mag])
            pos_ord = np.argsort(magnitude[pos_mag])

            if len(neg_ord) > 0:
                neg_mags = magnitude[neg_mag][neg_ord]
                neg_res = self.reoptimize_with_force(
                    modes,
                    magnitude=0,
                    force_scan_steps=neg_mags,
                    units=units,
                    use_internals=use_internals,
                    mass_weight=mass_weight,
                    optimizer_mode=optimizer_mode,
                    optimizer_method=optimizer_method,
                    profile_generator=profile_generator,
                    climb=climb,
                    ts_opt_settings=ts_opt_settings,
                    max_iterations=max_iterations,
                    max_displacement=max_displacement,
                    predistort=predistort,
                    reoptimize_reactants=reoptimize_reactants,
                    reoptimize_ts=reoptimize_ts,
                    initial_ts_step=initial_ts_step,
                    num_ts_steps=num_ts_steps,
                    initial_reactants_step=initial_reactants_step,
                    modify_forces=modify_forces,
                    apply_constraints=apply_constraints,
                    reembed_displacements=reembed_displacements,
                    remove_transrot=remove_transrot,
                    remove_orientation=remove_orientation,
                    output_dir=output_dir,
                    info_file='force_modified_{mode}_{mag}_neg.json',
                    displacements=displacements,
                    verbose=verbose,
                    run_gc=run_gc,
                    logger=logger,
                    rigid_max_recursion=rigid_max_recursion,
                    rigid_scan_steps=rigid_scan_steps,
                    rigid_scan_options=rigid_scan_options,
                    ts_initial_coords=ts_initial_coords,
                    rs_initial_coords=rs_initial_coords,
                    flatten_res=False,
                    # displacement_generator=None,
                    **opts
                )
            else:
                neg_res = []

            if len(pos_ord) > 0:
                pos_mags = magnitude[pos_mag][pos_ord]
                pos_res = self.reoptimize_with_force(
                    modes,
                    magnitude=0,
                    force_scan_steps=pos_mags,
                    units=units,
                    use_internals=use_internals,
                    mass_weight=mass_weight,
                    optimizer_mode=optimizer_mode,
                    optimizer_method=optimizer_method,
                    profile_generator=profile_generator,
                    climb=climb,
                    ts_opt_settings=ts_opt_settings,
                    max_iterations=max_iterations,
                    max_displacement=max_displacement,
                    predistort=predistort,
                    reoptimize_reactants=reoptimize_reactants,
                    reoptimize_ts=reoptimize_ts,
                    initial_ts_step=initial_ts_step,
                    num_ts_steps=num_ts_steps,
                    initial_reactants_step=initial_reactants_step,
                    modify_forces=modify_forces,
                    apply_constraints=apply_constraints,
                    reembed_displacements=reembed_displacements,
                    remove_transrot=remove_transrot,
                    remove_orientation=remove_orientation,
                    output_dir=output_dir,
                    info_file='force_modified_{mode}_{mag}_neg.json',
                    displacements=displacements,
                    verbose=verbose,
                    run_gc=run_gc,
                    logger=logger,
                    rigid_max_recursion=rigid_max_recursion,
                    rigid_scan_steps=rigid_scan_steps,
                    rigid_scan_options=rigid_scan_options,
                    ts_initial_coords=ts_initial_coords,
                    rs_initial_coords=rs_initial_coords,
                    flatten_res=False,
                    # displacement_generator=None,
                    **opts
                )
            else:
                pos_res = []

            # have to unwrap these results
            res = [
                [None] * len(magnitude)
                for _ in modes
            ]

            neg_inv = np.argsort(neg_ord)
            for i,(rs_i, ts_i, res_block) in enumerate(neg_res):
                res_block: ForceModifiedReactionData
                subterms = res_block.predistorted_data
                for j,t in enumerate(subterms):
                    t: ForceModifiedReactionData
                    j = neg_mag[0][neg_inv[j]]
                    res[i][j] = (
                        rs_i.modify(coords=t.force_modified_reactant_geom),
                        ts_i.modify(coords=t.force_modified_transition_state_geom),
                        t
                    )

            pos_inv = np.argsort(pos_ord)
            for i, (rs_i, ts_i, res_block) in enumerate(pos_res):
                res_block: ForceModifiedReactionData
                subterms = res_block.predistorted_data
                for j, t in enumerate(subterms):
                    t: ForceModifiedReactionData
                    j = pos_mag[0][pos_inv[j]]
                    res[i][j] = (
                        rs_i.modify(coords=t.force_modified_reactant_geom),
                        ts_i.modify(coords=t.force_modified_transition_state_geom),
                        t
                    )

        else:
            if units is not None:
                if isinstance(units, str):
                    units = units.replace("Newtons", "Joules/Meters")
                    units = units.split("/")
                conv = UnitsData.convert(units[0], "Hartrees") / (
                    UnitsData.convert(units[1], "BohrRadius")
                )
            else:
                conv = 1

            rs_init = self.rs
            if rs_initial_coords is not None:
                rs_init = rs_init.modify(coords=rs_initial_coords)
            ts_init = self.ts
            if ts_initial_coords is not None:
                ts_init = ts_init.modify(coords=ts_initial_coords)

            res = []
            if verbose:
                print(f"Optimizing forces over modes {modes} and magnitudes {mags}")
            for mode in modes:
                subres = []
                for magnitude in mags:
                    magnitude = conv * magnitude

                    if (not rigid) and (force_scan_steps is not None):
                        if nput.is_int(force_scan_steps):
                            fms = np.linspace(0, magnitude / conv, force_scan_steps)
                        else:
                            fms = np.asanyarray(force_scan_steps)
                        fmrd_steps = []
                        for f in fms:
                            rs_init, ts_init, fmrd = self.reoptimize_with_force(
                                mode,
                                magnitude=f,
                                units=units,
                                use_internals=use_internals,
                                mass_weight=mass_weight,
                                optimizer_mode=optimizer_mode,
                                optimizer_method=optimizer_method,
                                profile_generator=profile_generator,
                                climb=climb,
                                ts_opt_settings=ts_opt_settings,
                                max_iterations=max_iterations,
                                max_displacement=max_displacement,
                                predistort=predistort,
                                reoptimize_reactants=reoptimize_reactants,
                                reoptimize_ts=reoptimize_ts,
                                initial_ts_step=initial_ts_step,
                                num_ts_steps=num_ts_steps,
                                initial_reactants_step=initial_reactants_step,
                                modify_forces=modify_forces,
                                apply_constraints=apply_constraints,
                                reembed_displacements=reembed_displacements,
                                remove_transrot=remove_transrot,
                                remove_orientation=remove_orientation,
                                output_dir=output_dir,
                                info_file='force_modified_{mode}_{mag}_step.json',
                                displacements=displacements,
                                verbose=verbose,
                                run_gc=run_gc,
                                logger=logger,
                                rigid_max_recursion=rigid_max_recursion,
                                rigid_scan_steps=rigid_scan_steps,
                                rigid_scan_options=rigid_scan_options,
                                ts_initial_coords=ts_init.coords,
                                rs_initial_coords=rs_init.coords,
                                # displacement_generator=None,
                                **opts
                            )
                            fmrd_steps.append(fmrd)

                        fmrd = fmrd_steps[-1]
                        fmrd = fmrd._replace(predistorted_data=fmrd_steps)

                        r = rs_init.modify(coords=fmrd.force_modified_reactant_geom)
                        ts = ts_init.modify(coords=fmrd.force_modified_transition_state_geom)
                    else:
                        if rigid:
                            _, force_vector = self.mode_force_function(mode,
                                                                       magnitude=magnitude,
                                                                       displacements=displacements,
                                                                       use_internals=use_internals,
                                                                       mass_weight=mass_weight,
                                                                       reembed_displacements=reembed_displacements,
                                                                       remove_transrot=remove_transrot,
                                                                       remove_orientation=remove_orientation)
                            if rigid_scan_options is None:
                                rigid_scan_options = {}
                            if rigid_scan_steps is not None:
                                rigid_scan_options['steps'] = rigid_scan_steps
                            _, (dr, dt), _ = self.predicted_delta_from_forces(
                                mode,
                                magnitude,
                                displacements=displacements,
                                use_internals=use_internals,
                                mass_weight=mass_weight,
                                max_recursion=rigid_max_recursion,
                                return_geometries=True,
                                ts_initial_coords=ts_initial_coords,
                                rs_initial_coords=rs_initial_coords,
                                **rigid_scan_options
                            )

                            _, xr = dr[1]
                            r = rs_init.modify(coords=xr)
                            _, xt = dt[1]
                            ts = ts_init.modify(coords=xt)
                        else:
                            if predistort:
                                predistorted_data:ForceModifiedReactionData = self.reoptimize_with_force(
                                    mode,
                                    magnitude=magnitude / conv,
                                    units=units,
                                    use_internals=use_internals,
                                    mass_weight=mass_weight,
                                    optimizer_mode=optimizer_mode,
                                    optimizer_method=optimizer_method,
                                    profile_generator=profile_generator,
                                    climb=climb,
                                    ts_opt_settings=ts_opt_settings,
                                    max_iterations=max_iterations,
                                    max_displacement=max_displacement,
                                    predistort=False,
                                    reoptimize_reactants=reoptimize_reactants,
                                    reoptimize_ts=reoptimize_ts,
                                    initial_ts_step=initial_ts_step,
                                    num_ts_steps=num_ts_steps,
                                    initial_reactants_step=initial_reactants_step,
                                    modify_forces=modify_forces,
                                    apply_constraints=apply_constraints,
                                    reembed_displacements=reembed_displacements,
                                    remove_transrot=remove_transrot,
                                    remove_orientation=remove_orientation,
                                    output_dir=output_dir,
                                    info_file='force_modified_{mode}_{mag}_pre.json',
                                    displacements=displacements,
                                    verbose=verbose,
                                    run_gc=run_gc,
                                    logger=logger,
                                    rigid=True,
                                    rigid_max_recursion=rigid_max_recursion,
                                    rigid_scan_steps=rigid_scan_steps,
                                    rigid_scan_options=rigid_scan_options,
                                    ts_initial_coords=ts_initial_coords,
                                    rs_initial_coords=rs_initial_coords,
                                    # displacement_generator=None,
                                    **opts
                                )[2]

                                fmrd:ForceModifiedReactionData = self.reoptimize_with_force(
                                    mode,
                                    magnitude=magnitude,
                                    units='Hartrees/BohrRadius',
                                    use_internals=use_internals,
                                    mass_weight=mass_weight,
                                    optimizer_mode=optimizer_mode,
                                    optimizer_method=optimizer_method,
                                    profile_generator=profile_generator,
                                    climb=climb,
                                    ts_opt_settings=ts_opt_settings,
                                    max_iterations=max_iterations,
                                    max_displacement=max_displacement,
                                    predistort=False,
                                    reoptimize_reactants=reoptimize_reactants,
                                    reoptimize_ts=reoptimize_ts,
                                    initial_ts_step=initial_ts_step,
                                    num_ts_steps=num_ts_steps,
                                    initial_reactants_step=initial_reactants_step,
                                    modify_forces=modify_forces,
                                    apply_constraints=apply_constraints,
                                    reembed_displacements=reembed_displacements,
                                    remove_transrot=remove_transrot,
                                    remove_orientation=remove_orientation,
                                    output_dir=output_dir,
                                    info_file='force_modified_{mode}_{mag}_pre.json',
                                    displacements=displacements,
                                    verbose=verbose,
                                    run_gc=run_gc,
                                    logger=logger,
                                    rigid=False,
                                    rigid_max_recursion=rigid_max_recursion,
                                    rigid_scan_steps=rigid_scan_steps,
                                    rigid_scan_options=rigid_scan_options,
                                    ts_initial_coords=predistorted_data.force_modified_transition_state_geom,
                                    rs_initial_coords=predistorted_data.force_modified_reactant_geom,
                                    # displacement_generator=None,
                                    **opts
                                )[2]



                                fmrd = fmrd._replace(
                                    predistorted_data=predistorted_data
                                )

                                r = rs_init.modify(coords=fmrd.force_modified_reactant_geom)
                                ts = ts_init.modify(coords=fmrd.force_modified_transition_state_geom)
                                return [r, ts, fmrd]

                            if modify_forces:
                                if remove_orientation is None:
                                    remove_orientation = not apply_constraints
                                if remove_transrot is None:
                                    remove_transrot = not apply_constraints
                                gradient_modification_function, force_vector = self.mode_force_function(mode,
                                                                                                        magnitude=magnitude,
                                                                                                        displacements=displacements,
                                                                                                        use_internals=use_internals,
                                                                                                        mass_weight=mass_weight,
                                                                                                        reembed_displacements=reembed_displacements,
                                                                                                        remove_transrot=remove_transrot,
                                                                                                        remove_orientation=remove_orientation)
                            else:
                                gradient_modification_function, force_vector = None, None

                            if reoptimize_ts:
                                if profile_generator == 'relaxed':
                                    if ts_opt_settings is None:
                                        ts_opt_settings = {}
                                    if optimizer_method is not None:
                                        ts_opt_settings['method'] = ts_opt_settings.get('method', optimizer_method)

                                    ts_opt_settings = dict(
                                        max_iterations=max_iterations,
                                        max_displacement=max_displacement,
                                        logger=logger
                                    ) | (
                                                          dict(coordinate_constraints=[
                                                                  (0, 2),
                                                                  (1, 3)
                                                              ]) if apply_constraints else {}
                                                      ) | ts_opt_settings

                                    def pre_displace(coords):
                                        dd = self.get_displacement_dirs(
                                            mass_weight=mass_weight,
                                            use_internals=use_internals,
                                            displacements=displacements
                                        )
                                        d = dd[mode] * initial_reactants_step
                                        if use_internals:
                                            dx = self.internal_mols[1].get_cartesians_by_internals(1, coords=coords, strip_embedding=True)[0]
                                            d = np.dot(d, dx)
                                        return coords + d.reshape(-1, 3)

                                    ts = ts_init.optimize(gradient_modification_function=gradient_modification_function,
                                                          max_iterations=max_iterations,
                                                          mode=optimizer_mode,
                                                          initialization_function=pre_displace,
                                                          # logger=True,
                                                          **ts_opt_settings)
                                else:
                                    disp_t = ts_init.get_scan_coordinates(
                                        [[-initial_ts_step, initial_ts_step, num_ts_steps]],
                                        which=[0],
                                        coordinate_expansion=[self.ts.get_normal_modes().coords_by_modes],
                                        # internals='reembed' if use_internals else False,
                                        # strip_embedding=True if use_internals else False
                                    )

                                    if ts_opt_settings is None:
                                        ts_opt_settings = {}

                                    ts_opt_settings = dict(
                                        max_iterations=max_iterations,
                                        max_displacement=max_displacement,
                                        logger=logger
                                    ) | (
                                                          dict(coordinate_constraints=[
                                                              (0, 2),
                                                              (1, 3)
                                                          ]) if apply_constraints else {}
                                                      ) | ts_opt_settings

                                    images = [ts_init.modify(coords=t) for t in disp_t]
                                    rxn = Reaction(
                                        [images[0]],
                                        [images[-1]],
                                        optimize=False,
                                        align=False
                                    )

                                    if 'dimer' in profile_generator:
                                        ts_opt_settings['image_guess'] = ts_opt_settings.get('image_guess', 0)
                                        images = [images[0], images[-1]]
                                    if profile_generator == 'ase-dimer':
                                        ts_opt_settings['method_options'] = {
                                                                                'image_guess': ts_opt_settings.pop('image_guess', 0)
                                                                            } | ts_opt_settings.get('method_options', {})
                                    prof = rxn.get_profile_generator(profile_generator,
                                                                     climb=climb,
                                                                     energy_evaluator=self.ts.energy_evaluator)

                                    new_images = prof.generate(base_images=[images[0], images[-1]],
                                                               gradient_modification_function=gradient_modification_function,
                                                               **ts_opt_settings)
                                    ts = new_images[0]
                            else:
                                ts = ts_init

                            if reoptimize_reactants:
                                if optimizer_method is not None:
                                    opts['method'] = opts.get('method', optimizer_method)
                                if apply_constraints:
                                    opts['coordinate_constraints'] = [
                                        (0, 2),
                                        (1, 3)
                                    ]

                                def pre_displace(coords):
                                    dd = self.get_displacement_dirs(
                                        mass_weight=mass_weight,
                                        use_internals=use_internals,
                                        displacements=displacements
                                    )
                                    if not callable(dd):
                                        d = dd[mode] * initial_reactants_step
                                    else:
                                        d = dd(self.ts, coords)[mode] * initial_reactants_step
                                    if use_internals:
                                        dx = self.internal_mols[0].get_cartesians_by_internals(1, coords=coords, strip_embedding=True)[0]
                                        d = np.dot(d, dx)
                                    return coords + d.reshape(-1, 3)

                                r = self.rs.modify(internals=None).optimize(
                                    gradient_modification_function=gradient_modification_function,
                                    max_iterations=max_iterations,
                                    mode=optimizer_mode,
                                    initialization_function=pre_displace,
                                    max_displacement=max_displacement,
                                    logger=logger,
                                    **opts)
                            else:
                                r = self.rs

                        fmrd = ForceModifiedReactionData(
                            atoms=self.ts.atoms,
                            reactant_geom=self.rs.coords,
                            transition_state_geom=self.ts.coords,
                            reactant_energy=self.rs.calculate_energy(),
                            transition_state_energy=self.ts.calculate_energy(),
                            force_modified_reactant_geom=r.coords,
                            force_modified_transition_state_geom=ts.coords,
                            force_modified_reactant_energy=r.calculate_energy(),
                            force_modified_transition_state_energy=ts.calculate_energy(),
                            force_vector=force_vector,
                            force_magnitude=magnitude,
                            force_units=units,
                            mass_weight=mass_weight,
                            energy_evaluator=self.rs.energy_evaluator,
                            internals=self.internals,
                            optimizer_settings=dict(
                                optimizer_mode=optimizer_mode,
                                optimizer_method=optimizer_method,
                                profile_generator=profile_generator,
                                ts_opt_settings=ts_opt_settings,
                                max_iterations=max_iterations,
                                initial_ts_step=initial_ts_step,
                                num_ts_steps=num_ts_steps,
                                initial_reactants_step=initial_reactants_step
                            ) | opts
                        )

                        if output_dir is not None:
                            conv2 = UnitsData.convert("Hartrees", "Picojoules") / (
                                UnitsData.convert("BohrRadius", "Meters")
                            )
                            m = np.round(magnitude * conv2)
                            if os.path.splitext(output_dir)[-1].startswith('.'):
                                output_dir, info_file = os.path.split(output_dir)
                            output_dir = output_dir.format(mode=mode, mag=m)
                            info_file = info_file.format(mode=mode, mag=m)
                            if len(output_dir) > 0:
                                os.makedirs(output_dir, exist_ok=True)
                                info_file = os.path.join(output_dir, info_file)
                            utils.write_namedtuple(
                                info_file,
                                fmrd
                            )

                    if run_gc: gc.collect()
                    subres.append([r, ts, fmrd])
                res.append(subres)

        if flatten_res:
            res = sum(res, [])
            if smol_mode and smol_force:
                res = res[0]
        elif smol_mode:
            res = res[0]
            if smol_force:
                res = res[0]
        elif smol_force:
            res = [r[0] for r in res]

        return res

    @property
    def _internals_selectors(self):
        return {
            'dihedrals':self._select_dihedrals,
            'angles':self._select_angles,
            'bonds':self._select_bonds,
        }
    @classmethod
    def _select_by_filter(cls, internals, *, filter, optimizer, fragment_indices=None):
        base_sel = [i for i,c in enumerate(internals) if filter(c)]
        if fragment_indices is not None:
            if nput.is_int(fragment_indices):
                fragment_indices = optimizer.rs.fragment_indices[fragment_indices]
            base_sel = [b for b in base_sel if all(i in fragment_indices for i in internals[b])]
        return base_sel
    @classmethod
    def _select_dihedrals(cls, internals, *, optimizer, fragment_indices=None):
        return cls._select_by_filter(internals, filter=lambda x:len(x)==4, optimizer=optimizer, fragment_indices=fragment_indices)
    @classmethod
    def _select_bonds(cls, internals, *, optimizer, fragment_indices=None):
        return cls._select_by_filter(internals, filter=lambda x:len(x)==2, optimizer=optimizer, fragment_indices=fragment_indices)
    @classmethod
    def _select_angles(cls, internals, *, optimizer, fragment_indices=None):
        return cls._select_by_filter(internals, filter=lambda x:len(x)==3, optimizer=optimizer, fragment_indices=fragment_indices)

    def compute_gammas(self, which=None, mass_weight=True, displacements=None, use_internals=False,
                       return_components=False):
        dd = self.get_displacement_dirs(
            mass_weight=mass_weight,
            use_internals=use_internals,
            displacements=displacements
        )
        if which is not None:
            d = dd[which,]
        else:
            d = dd
        if use_internals:
            dx_ts = self.internal_mols[1].get_cartesians_by_internals(1, strip_embedding=True)[0]
            direction_ts = np.dot(d, dx_ts)

            dx_rs = self.internal_mols[0].get_cartesians_by_internals(1, strip_embedding=True)[0]
            dirs = np.dot(d, dx_rs)
        else:
            dirs = direction_ts = d

        return compute_reaction_gamma(self.rs, self.ts,
                                      dirs,
                                      direction_ts=direction_ts,
                                      use_mode_space=self.use_mode_space,
                                      return_components=return_components
                                      # modes=self.prepped_modes
                                      )
    def get_selected_internals(self, which,
                               max_internals=None,
                               max_internals_ranks=None,
                               fragment_indices=None,
                               lookup_internals_index=None,
                               use_internals=True,
                               displacements=None,
                               mass_weight=False,
                               return_gammas=False):
        if displacements is None:
            displacements = self.pure_internal_displacement_matrix
        if isinstance(which, str) or callable(which):
            which = {
                'selector':which
            }
        if isinstance(which, dict):
            which = which.copy()
            if fragment_indices is None:
                fragment_indices = self.opts.get('fragment_indices')
            if fragment_indices is not None:
                which['fragment_indices'] = fragment_indices
            selector = which.pop('selector')
            if isinstance(selector, str):
                selector = self._internals_selectors[selector]
            which = selector(coordops.extract_zmatrix_internals(self.internals, strip_embedding=True),
                             optimizer=self,
                             **which)
            if lookup_internals_index is None:
                lookup_internals_index = False
        elif lookup_internals_index is None:
            lookup_internals_index = True
        if not nput.is_int(which) and lookup_internals_index:
            which = coordops.zmatrix_indices(self.internals, which, strip_embedding=True)
        if not nput.is_int(which) and len(which) == 0:
            if return_gammas:
                return which, []
            else:
                return which
        if max_internals is not None and not nput.is_int(which):
            which = np.asanyarray(which)
            gammas = self.compute_gammas(which,
                                         use_internals=use_internals,
                                         displacements=displacements,
                                         mass_weight=mass_weight)
            if len(gammas) > max_internals:
                sel = np.argpartition(gammas, -(max_internals+1))[-max_internals:]
            else:
                sel = np.argsort(-gammas)
            gammas = gammas[sel,]
            which = which[sel,]
            if max_internals_ranks is not None:
                gammas = gammas[max_internals_ranks,]
                which = which[max_internals_ranks,]
        else:
            gammas = None

        if return_gammas:
            return which, gammas
        else:
            return which

    def generate_internal_distortions(self,
                                      which,
                                      mass_weight=False,
                                      max_internals=None,
                                      max_internals_ranks=None,
                                      displacements=None,
                                      use_internals=True,
                                      lookup_internals_index=None,
                                      fragment_indices=None,
                                      **etc
                                      ):
        if displacements is None:
            displacements = self.pure_internal_displacement_matrix
        which = self.get_selected_internals(which,
                                            lookup_internals_index=lookup_internals_index,
                                            max_internals=max_internals,
                                            max_internals_ranks=max_internals_ranks,
                                            fragment_indices=fragment_indices,
                                            displacements=displacements,
                                            use_internals=use_internals,
                                            return_gammas=False)[0]
        return self.get_displaced_geometries(mode=which,
                                             mass_weight=mass_weight,
                                             displacements=displacements,
                                             use_internals=use_internals,
                                             **etc)

    def reoptimize_internals_with_force(self,
                                        which,
                                        magnitude=50,
                                        mass_weight=False,
                                        max_internals=None,
                                        max_internals_ranks=None,
                                        units='PicoJoules/Meters',
                                        displacements=None,
                                        use_internals=True,
                                        lookup_internals_index=None,
                                        fragment_indices=None,
                                        verbose=False,
                                        return_selected=False,
                                        **opts
                                        ):
        if displacements is None:
            displacements = self.pure_internal_displacement_matrix
        which, gammas = self.get_selected_internals(which,
                                                    lookup_internals_index=lookup_internals_index,
                                                    max_internals=max_internals,
                                                    max_internals_ranks=max_internals_ranks,
                                                    fragment_indices=fragment_indices,
                                                    displacements=displacements,
                                                    use_internals=use_internals,
                                                    return_gammas=True)
        fmrds = self.reoptimize_with_force(
            which,
            mass_weight=mass_weight,
            magnitude=magnitude,
            units=units,
            displacements=displacements,
            use_internals=use_internals,
            verbose=verbose,
            **opts
        )
        if return_selected:
            return fmrds, (which, gammas)
        else:
            return fmrds
    # def reoptimize
    def _hcff_pressure(self, ts: Molecule, coords, *, pressure, surface_points=500, radius_scaling=1):
        coords = np.asanyarray(coords).reshape((-1, 3))
        surf = ts.modify(coords=coords).get_surface(
            samples=surface_points,
            radius_scaling=radius_scaling)
        area = surf.surface_area(method='sampling')
        fmax = area * pressure
        centroid = np.average(coords, axis=0)
        normals = centroid[np.newaxis] - coords
        dists = np.linalg.norm(normals, axis=1)
        normals = normals / dists[:, np.newaxis]
        dist_fractions = dists / np.max(dists)
        scaled_normals = dist_fractions[..., np.newaxis] * normals * fmax
        return scaled_normals.reshape((1, -1))
    def _xhcff_pressure(self, ts: Molecule, coords, *, pressure, surface_points=500, radius_scaling=1.2):
        coords = np.asanyarray(coords).reshape((-1, 3))
        surf = ts.modify(coords=coords).get_surface(samples=surface_points,
                                                    radius_scaling=radius_scaling).get_triangulation()
        groups, _ = nput.group_by(np.arange(len(surf.tri_map)), surf.tri_map)
        areas = surf.surface_area(return_components=True)
        area_fractions = areas #/ np.sum(areas)
        normals = -surf.normals
        # surf.plot(solid=True, normals=True, normal_scaling=.1).show()
        scaled_normals = area_fractions[..., np.newaxis] * normals * pressure
        force = np.zeros_like(coords)
        for atom, inds in zip(*groups):
            force[atom] = np.sum(scaled_normals[inds,], axis=0)
        return force.reshape((1, -1))
    def _cylinder_pressure(self, ts: Molecule, coords, *, pressure,
                           axis, radius=1, centroid=None,
                           surface_points=200, radius_scaling=1, bidirectional=False):
        coords = np.asanyarray(coords).reshape((-1, 3))
        mol = ts.modify(coords=coords)
        surf = mol.get_surface(samples=surface_points, radius_scaling=radius_scaling)
        atom_areas = 4 * np.pi * surf.radii**2
        if centroid is None:
            centroid = mol.center_of_mass
        if callable(centroid):
            centroid = centroid(coords)
        else:
            centroid = np.asanyarray(centroid)
        if callable(axis):
            axis = axis(coords, centroid)
        else:
            axis = nput.vec_normalize(axis) #TODO: do outside loop for a quick opt

        atom_points = surf.atom_sampling_points
        # ids = np.concatenate([[i] * len(p) for i,p in enumerate(atom_points)])
        shifted_surf_ponts = np.concatenate(atom_points, axis=0) - centroid[np.newaxis, :]
        axis_projection = shifted_surf_ponts @ axis[:, np.newaxis]
        axis_distance = nput.vec_norms(shifted_surf_ponts - axis_projection * axis[np.newaxis, :])
        if bidirectional:
            mask = (axis_distance < radius)
            max_dist = np.max(np.abs(axis_projection[mask]))
        else:
            mask = (axis_distance < radius) & (axis_projection.flatten() > 0)
            max_dist = np.max(axis_projection[mask])
        axis_projection = axis_projection / max_dist

        split_regions = np.cumsum([len(p) for p in atom_points])
        mask_inds = np.array_split(np.arange(split_regions[-1]), split_regions[:-1])
        force = np.zeros_like(coords)
        for atom, x_block in enumerate(mask_inds):
            # fraction of the total surface
            # scaled by atom contrib to total
            nt = np.sum(mask[x_block,])
            if nt == 0: continue
            mask_block = mask[x_block,]
            frac = nt / len(mask_inds)
            axis_term = axis_projection[x_block,][mask_block]
            contrib = axis * atom_areas[atom] * frac * pressure * np.average(axis_term)
            force[atom] = -contrib
        return force.reshape((1, -1))
    def _cavity_pressure(self, ts: Molecule, coords, *, pressure, surface_points=100, radius_scaling=1.6):
        coords = np.asanyarray(coords).reshape((-1, 3))
        surf = ts.modify(coords=coords).get_surface(samples=surface_points,
                                                    radius_scaling=radius_scaling).get_triangulation()
        groups, _ = nput.group_by(np.arange(len(surf.tri_map)), surf.tri_map)
        # tri_inds = surf.inds
        _, derivs = surf.volume_derivatives(return_components=True)
        # areas = surf.surface_area(return_components=True)
        # area_fractions = areas #/ np.sum(areas)
        derivs = derivs.reshape((derivs.shape[0], -1, 3)) * pressure
        force = np.zeros_like(coords)
        for atom, inds in zip(*groups):
            force[atom] += np.sum(np.sum(derivs[inds,], axis=0), axis=0)
        return force.reshape((1, -1))
    def pressure_model_generator(self, pressure_model, pressure, **opts):
        if dev.str_is(pressure_model, 'hcff'):
            pressure_model = self._hcff_pressure
        elif dev.str_is(pressure_model, 'xhcff'):
            pressure_model = self._xhcff_pressure
        elif dev.str_is(pressure_model, 'cavity'):
            pressure_model = self._cavity_pressure
        elif dev.str_is(pressure_model, 'cylinder'):
            if opts.get('axis') is None:
                opts['axis'] = 'c'
            _, axes = nput.moments_of_inertia(self.ts.coords, self.ts.atomic_masses)
            if dev.str_is(opts['axis'], 'c'):
                opts['axis'] = axes[:, 2]
            elif dev.str_is(opts['axis'], '-c'):
                opts['axis'] = -axes[:, 2]
            elif dev.str_is(opts['axis'], 'a'):
                opts['axis'] = axes[:, 0]
            elif dev.str_is(opts['axis'], '-a'):
                opts['axis'] = -axes[:, 0]
            elif dev.str_is(opts['axis'], 'b'):
                opts['axis'] = axes[:, 1]
            elif dev.str_is(opts['axis'], '-b'):
                opts['axis'] = -axes[:, 1]
            elif isinstance(opts['axis'], str):
                raise NotImplementedError(f"can't handle axis '{opts['axis']}'")
            pressure_model = self._cylinder_pressure
        elif not callable(pressure_model):
            raise NotImplementedError(f"pressure model `{pressure_model}` not implemented")

        def apply_pressure(ref, coords):
            return pressure_model(ref, coords, pressure=pressure, **opts)
        return apply_pressure

    def get_pressure_distorted_geometries(self,
                                          which=0,
                                          disp_min=0, disp_max=5,
                                          mass_weight=False,
                                          pressure_model='xhcff',
                                          pressure_options=None,
                                          **etc):
        if pressure_options is None:
            pressure_options = {}
        displacements = self.pressure_model_generator(pressure_model,
                                                      pressure=1,
                                                      **pressure_options)
        return self.get_displaced_geometries(mode=which,
                                             disp_min=disp_min, disp_max=disp_max,
                                             mass_weight=mass_weight,
                                             displacements=displacements, **etc)

    def get_pressure_distortion_energies(self,
                                         which=0,
                                         disp_min=0, disp_max=5,
                                         mass_weight=False,
                                         pressure_model='xhcff',
                                         pressure_options=None,
                                         **etc):
        if pressure_options is None:
            pressure_options = {}
        displacements = self.pressure_model_generator(pressure_model,
                                                      pressure=1,
                                                      **pressure_options)
        return self.get_distortion_energies(mode=which,
                                            disp_min=disp_min, disp_max=disp_max,
                                            mass_weight=mass_weight,
                                            displacements=displacements, **etc)

    default_pressure = 200
    def reoptimize_with_pressure(self,
                                 magnitude=None,
                                 pressure_units="Megapascals",
                                 *,
                                 pressure=None,
                                 which=0,
                                 pressure_model='xhcff',
                                 pressure_options=None,
                                 apply_constraints=False,
                                 remove_orientation=False,
                                 remove_transrot=False,
                                 displacements=None,
                                 **etc
                                 ):
        if pressure is not None:
            if magnitude is not None:
                raise ValueError("can't get both `magnitude` and `pressure` keywords (they are synonyms)")
            magnitude = pressure
        if isinstance(pressure_units, str):
            pressure_units = pressure_units.replace("pascals", "Pascals").replace("Pascals", "Newtons/MetersSquared")
            pressure_units = pressure_units.rsplit("/", 1)
        force_units, area_units = pressure_units
        area_scaling = UnitsData.convert(area_units, "BohrRadiusSquared")
        if isinstance(force_units, str):
            force_units = force_units.replace("Newtons", "Joules/Meters")
            force_units = force_units.split("/")
        # all of this is to make unit debugging a little easier, did it actually? I don't know
        conv = UnitsData.convert(force_units[0], "Hartrees") / (
            UnitsData.convert(force_units[1], "BohrRadius")
        )
        pressure_scaling = conv / area_scaling
        force_units = "Hartrees/BohrRadius"
        if displacements is None:
            if pressure_options is None:
                pressure_options = {}
            displacements = self.pressure_model_generator(pressure_model,
                                                          pressure=pressure_scaling,
                                                          **pressure_options
                                                          )
        else:
            magnitude = np.asanyarray(magnitude) * pressure_scaling
        return self.reoptimize_with_force(
            which,
            magnitude=magnitude,
            displacements=displacements,
            units=force_units,
            remove_orientation=remove_orientation,
            remove_transrot=remove_transrot,
            apply_constraints=apply_constraints,
            **etc
        )

    def reoptimize_with_random_force(self,
                                     which,
                                     magnitude=50,
                                     mass_weight=False,
                                     units='PicoJoules/Meters',
                                     displacements=None,
                                     use_internals=True,
                                     verbose=False,
                                     **opts
                                     ):
        if displacements is None:
            displacements = self.random_dirs
        fmrds = self.reoptimize_with_force(
            which,
            mass_weight=mass_weight,
            magnitude=magnitude,
            units=units,
            displacements=displacements,
            use_internals=use_internals,
            verbose=verbose,
            **opts
        )
        return fmrds

    def get_distortion_steric_repulsions(self,
                                         mode=0,
                                         disp_min=0,
                                         disp_max=1,
                                         steps=5,
                                         mass_weight=False,
                                         prune=False,
                                         generate_displacement_function=None,
                                         density=10,
                                         pairwise_term=interior_repulsion,
                                         return_breakdowns=False,
                                         preserve_clipping=False,
                                         atom_groups=None,
                                         only_endpoints=False,
                                         **opts):
        if generate_displacement_function is None:
            generate_displacement_function = self.get_displaced_geometries
        (v, x_r, x_t) = generate_displacement_function(mode,
                                             disp_min=disp_min,
                                             disp_max=disp_max,
                                             mass_weight=mass_weight,
                                             steps=steps,
                                             **opts)
        if only_endpoints:
            x_r = x_r[(0, -1),]
            x_t = x_t[(0, -1),]
        rs_sterics = molecule_steric_potential(self.rs,
                                               molecule_distortion_function=lambda mol:x_r,
                                               density=density,
                                               prune=prune,
                                               pairwise_term=pairwise_term,
                                               atom_groups=atom_groups,
                                               preserve_clipping=preserve_clipping,
                                               return_breakdowns=return_breakdowns)
        ts_sterics = molecule_steric_potential(self.ts,
                                               molecule_distortion_function=lambda mol:x_t,
                                               density=density,
                                               prune=prune,
                                               pairwise_term=pairwise_term,
                                               atom_groups=atom_groups,
                                               preserve_clipping=preserve_clipping,
                                               return_breakdowns=return_breakdowns)
        if return_breakdowns:
            pts_r, pots_r = rs_sterics
            rs_sterics = (pts_r[1:], pots_r[1:])
            pts_t, pots_t = ts_sterics
            ts_sterics = (pts_t[1:], pots_t[1:])

        if disp_min < 0 and disp_max == 0:
            v = np.flip(v)
            x_r = np.flip(x_r, axis=0)
            x_t = np.flip(x_t, axis=0)
            if return_breakdowns:
                rs_sterics = tuple(list(reversed(r)) for r in rs_sterics)
                ts_sterics = tuple(list(reversed(t)) for t in ts_sterics)
            else:
                rs_sterics = np.flip(rs_sterics, axis=0)
                ts_sterics = np.flip(ts_sterics, axis=0)

        return (rs_sterics, ts_sterics), (v, x_r, x_t)

    def get_distortion_volume_changes(self,
                                      mode=0,
                                      disp_min=0,
                                      disp_max=1,
                                      steps=5,
                                      mass_weight=False,
                                      generate_displacement_function=None,
                                      volume_options=None,
                                      only_endpoints=False,
                                      **opts):
        if generate_displacement_function is None:
            generate_displacement_function = self.get_displaced_geometries
        (v, x_r, x_t) = generate_displacement_function(mode,
                                             disp_min=disp_min,
                                             disp_max=disp_max,
                                             mass_weight=mass_weight,
                                             steps=steps,
                                             **opts)
        if volume_options is None:
            volume_options = {}
        volume_options['use_mol_ref'] = volume_options.get('use_mol_ref', False)

        if only_endpoints:
            x_r = x_r[(0, -1),]
            x_t = x_t[(0, -1),]
        rs_volumes = molecule_volume_change(self.rs, molecule_distortion_function=lambda mol:x_r, **volume_options)
        ts_volumes = molecule_volume_change(self.ts, molecule_distortion_function=lambda mol:x_t, **volume_options)

        if disp_min < 0 and disp_max == 0:
            # v = np.flip(v)
            x_r = np.flip(x_r, axis=0)
            x_t = np.flip(x_t, axis=0)
            rs_volumes = np.flip(rs_volumes, axis=0)
            ts_volumes = np.flip(ts_volumes, axis=0)
            rs_volumes = rs_volumes - rs_volumes[0]
            ts_volumes = ts_volumes - ts_volumes[0]
        elif disp_min == 0:
            rs_volumes = rs_volumes - rs_volumes[0]
            ts_volumes = ts_volumes - ts_volumes[0]

        return (rs_volumes, ts_volumes), (v, x_r, x_t)

    def direction_overlap(self, other, mol='ts'):
        fds = self.force_dirs
        if isinstance(fds, np.ndarray):
            fds = (fds, fds)
        fds2 = other.force_dirs_inverse
        if isinstance(fds2, np.ndarray):
            fds2 = (fds2, fds2)
        if mol == 'ts':
            return fds[1] @ fds2[1]
        else:
            return fds[0] @ fds2[0]

    def mode_overlap(self, mol='ts'):
        fds = self.force_dirs
        if isinstance(fds, np.ndarray):
            fds = (fds, fds)
        if mol == 'ts':
            return fds[1] @ self.ts_modes.modes_by_coords
        else:
            return fds[0] @ self.rs_modes.modes_by_coords

    @classmethod
    def plot_overlap(self, overlap, **styles):
        return plt.ArrayPlot(overlap**2, vmin=0, vmax=1, **styles)
    @classmethod
    def overlap_breakdown(self, overlap):
        return np.round(overlap**2 * 100)

    def animate_normed(self, i, expansion=None,
                       modes=None,
                       use_internals=False, mag=.5, mol='ts',
                       mass_weight=True,
                       displacements=None,
                       **opts):
        if expansion is None and not use_internals:
            expansion = self.force_dirs
        if expansion is not None:
            if dev.str_is(mol, 'ts'):
                mol = self.ts
                if len(expansion) == 2 and expansion[0].ndim == 2:
                    expansion = expansion[1]
            elif dev.str_is(mol, 'reactant'):
                mol = self.rs
                if len(expansion) == 2 and expansion[0].ndim == 2:
                    expansion = expansion[0]
        if displacements is None:
            exp = mass_weighted_normalize_displacements(mol, mass_weight=mass_weight,  expansion=expansion)
            return mol.animate_coordinate(i, mag,
                                          coordinate_expansion=[nput.vec_normalize(exp, axis=1)],
                                          **opts
                                          )
        else:
            (x, sr, st) = self.get_displaced_geometries(0,
                                                        disp_min=-1, disp_max=1,
                                                        steps=3,
                                                        mass_weight=mass_weight, use_internals=use_internals,
                                                        displacements=displacements)

            if mol == 'ts':
                return self.ts.plot(st, **opts)
            else:
                return self.rs.plot(sr, **opts)

    def animate_mode(self, i, modes=None, mol='ts', **opts):
        if dev.str_is(mol, 'ts'):
            mol = self.ts
            if modes is None:
                modes = self.ts_modes
        elif dev.str_is(mol, 'reactant'):
            mol = self.rs
            if modes is None:
                modes = self.rs_modes
        return mol.animate_mode(i, modes=modes, **opts)

    def _get_default_displacement_steps(self, mass_weight=True):
        if mass_weight:
            return [-50, 50]
        else:
            return [-.5, .5]
    def get_displacement_dirs(self,
                              displacements=None,
                              mass_weight=True,
                              use_internals=False):
        no_disp = displacements is None
        if not callable(displacements):
            if use_internals:
                if no_disp:
                    displacements = self.internal_dirs
                    ref_dirs = self.force_dirs
                else:
                    ref_dirs = displacements @ self.internal_mols[1].get_cartesians_by_internals(1, strip_embedding=True)[0]
                if not no_disp or not mass_weight:
                    _, norms = mass_weighted_normalize_displacements(self.ts,
                                                                     mass_weight=mass_weight,
                                                                     expansion=ref_dirs,
                                                                     return_norms=True)
                    displacements = displacements / norms[:, np.newaxis]
            else:
                if displacements is None:
                    displacements = self.force_dirs
                if not no_disp or not mass_weight:
                    displacements = mass_weighted_normalize_displacements(self.rs,
                                                                          mass_weight=mass_weight,
                                                                          expansion=displacements)
        else:
            ...
        return displacements
    def get_displaced_geometries(self, mode,
                                 disp_min=None, disp_max=None, steps=50,
                                 scan_values=None,
                                 mass_weight=True,
                                 displacements=None,
                                 ts_initial_coords=None,
                                 rs_initial_coords=None,
                                 use_internals=False):
        defaults = self._get_default_displacement_steps(mass_weight=mass_weight)
        if disp_min is None:
            disp_min = defaults[0]
        if disp_max is None:
            disp_max = defaults[1]

        displacements = self.get_displacement_dirs(mass_weight=mass_weight, use_internals=use_internals,
                                                   displacements=displacements)

        if not callable(displacements):
            if use_internals:
                r, t = self.internal_mols
            else:
                r, t = self.rs, self.ts
            
            if ts_initial_coords is not None:
                t = t.modify(coords=ts_initial_coords)
            if rs_initial_coords is not None:
                r = r.modify(coords=rs_initial_coords)

            if scan_values is None:
                scan_values = np.linspace(disp_min, disp_max, steps)
                scan_coords_r = r.get_scan_coordinates(
                    [[disp_min, disp_max, steps]],
                    which=[mode],
                    coordinate_expansion=[displacements],
                    internals='reembed' if use_internals else False,
                    strip_embedding=True if use_internals else False
                )
                scan_coords_t = t.get_scan_coordinates(
                    [[disp_min, disp_max, steps]],
                    which=[mode],
                    coordinate_expansion=[displacements],
                    internals='reembed' if use_internals else False,
                    strip_embedding=True if use_internals else False
                )
            else:
                scan_values = np.asanyarray(scan_values)
                if scan_values.ndim == 1: scan_values = scan_values[:, np.newaxis]
                scan_coords_r = r.get_displaced_coordinates(
                    scan_values,
                    which=[mode],
                    coordinate_expansion=[displacements],
                    use_internals='reembed' if use_internals else False,
                    strip_embedding=True if use_internals else False
                )
                scan_coords_t = t.get_displaced_coordinates(
                    scan_values,
                    which=[mode],
                    coordinate_expansion=[displacements],
                    use_internals='reembed' if use_internals else False,
                    strip_embedding=True if use_internals else False
                )
        else:
            svs = scan_values
            if scan_values is None:
                scan_values = [disp_min, disp_max, steps]
                svs = np.linspace(disp_min, disp_max, steps)
                absolute_mesh = False
            else:
                absolute_mesh = True
            def displacement_generator(coords, disp):
                d = displacements(self.rs, coords)
                d = self.get_displacement_dirs(mass_weight=mass_weight,
                                               use_internals=use_internals,
                                               displacements=d)
                return [d], [mode]
            _, scan_coords_r, _ = self.rs.relaxed_scan(
                [scan_values],
                displacement_generator,
                max_iterations=0,
                absolute_mesh=absolute_mesh
            )
            def displacement_generator(coords, disp):
                d = displacements(self.ts, coords)
                d = self.get_displacement_dirs(mass_weight=mass_weight,
                                               use_internals=use_internals,
                                               displacements=d)
                return [d], [mode]
            _, scan_coords_t, _ = self.rs.relaxed_scan(
                [scan_values],
                displacement_generator,
                max_iterations=0,
                absolute_mesh=absolute_mesh
            )
            scan_values = svs

        return scan_values, scan_coords_r, scan_coords_t

    def get_distortion_energies(self, mode, disp_min=None, disp_max=None, steps=50,
                                order=None,
                                shift=True,
                                use_internals=False,
                                displacements=None,
                                ts_initial_coords=None,
                                rs_initial_coords=None,
                                return_geometries=False,
                                mass_weight=True):
        x, sr, st = self.get_displaced_geometries(mode, disp_min=disp_min, disp_max=disp_max, steps=steps,
                                                  mass_weight=mass_weight, use_internals=use_internals,
                                                  ts_initial_coords=ts_initial_coords,
                                                  rs_initial_coords=rs_initial_coords,
                                                  displacements=displacements)
        if displacements is not None:
            displacements = self.get_displacement_dirs(mass_weight=mass_weight, use_internals=use_internals,
                                                       displacements=displacements)

        eng_r = self.rs.calculate_energy(coords=sr, order=order)
        eng_ts = self.ts.calculate_energy(coords=st, order=order)
        if order is None:
            if shift:
                eng_r, eng_ts = eng_r - np.min(eng_r), eng_ts - np.min(eng_ts)
        else:
            if shift:
                eng_r[0] = eng_r[0] - np.min(eng_r[0])
                eng_ts[0] = eng_ts[0] - np.min(eng_ts[0])
            if use_internals:
                ft = np.array([
                    self.internal_mols[1].get_cartesians_by_internals(1, coords=g)[0]
                    for g in st
                ])
                fr = np.array([
                    self.internal_mols[0].get_cartesians_by_internals(1, coords=g)[0]
                    for g in sr
                ])
                ecs = self.internal_mols[1].internal_coordinates.system.embedding_coords
                rem = np.delete(np.arange(ft.shape[1]), ecs)
                ft = ft[:, rem, :]
                fr = fr[:, rem, :]
                # print(ft.shape, self.internal_dirs.shape, fr.shape, rem.shape)
                if displacements is None:
                    disps_t = self.internal_dirs[np.newaxis] @ ft
                    disps_r = self.internal_dirs[np.newaxis] @ fr
                else:
                    disps_t = displacements[np.newaxis] @ ft
                    disps_r = displacements[np.newaxis] @ fr
            else:
                if displacements is None:
                    disps_r = disps_t = self.force_dirs
                else:
                    disps_r = disps_t = displacements

            eng_r = eng_r[:1] + nput.tensor_reexpand([disps_r], eng_r[1:], axes=[-1, -1])
            eng_ts = eng_ts[:1] + nput.tensor_reexpand([disps_t], eng_ts[1:], axes=[-1, -1])

        if return_geometries:
            return (x, eng_r, eng_ts), (sr, st)
        else:
            return x, eng_r, eng_ts

    @classmethod
    def plot_eng_comp(cls, x, eng_r, eng_ts, **opts):
        if not nput.is_numeric(eng_r[0]):
            eng_r = eng_r[0]
        if not nput.is_numeric(eng_ts[0]):
            eng_ts = eng_ts[0]
        return plt.plot_multi(
            {'y': eng_r * UnitsData.convert("Hartrees", "Kilocalories/Mole"), 'label': 'gs'},
            {'y': eng_ts * UnitsData.convert("Hartrees", "Kilocalories/Mole"), 'label': 'ts'},
            x=x,
            **collections.ChainMap(
                opts,
                dict(
                    plot_legend=True,
                    axes_labels=[r'x ($a_0\text{-ish}$)', r'$\Delta$E (kcal mol$^{-1})$'],
                    legend_style={
                        'frameon': False,
                        'fontsize': 13
                    },
                    image_size=500
                )
            )
        )

    @classmethod
    def plot_force_comp(cls, mode, x, exp_r, exp_t,
                        units=None,
                        force_unit='kcal mol$^{-1}$/a$_0$-ish',
                        **opts
                        ):
        e_r = exp_r[1][:, mode]
        e_t = exp_t[1][:, mode]
        if units is not None:
            if isinstance(units, str):
                units = units.split("/")
            conv = UnitsData.convert("Hartrees", units[0]) / (
                UnitsData.convert("BohrRadius", units[1])
            ) * UnitsData.convert("Kilocalories/Mole", "Hartrees") # cancels out `plot_eng_comp`
            e_r = e_r * conv
            e_t = e_t * conv

        return cls.plot_eng_comp(x,
                                 e_r,
                                 e_t,
                                 axes_labels=["x ($a_0$-ish)", fr"$\partial E/\partial x$ ({force_unit})"],
                                 **opts
                                 )
    def plot_distortion_energies(self, mode, disp_min=None, disp_max=None, steps=50, mass_weight=True,
                                 use_internals=False, displacements=None,
                                 **opts):
        x, eng_r, eng_ts = self.get_distortion_energies(mode, disp_min=disp_min, disp_max=disp_max, steps=steps,
                                                        use_internals=use_internals, displacements=displacements,
                                                        mass_weight=mass_weight)
        return self.plot_eng_comp(x, eng_r, eng_ts, **opts)

    def plot_distortion_forces(self, mode, disp_min=None, disp_max=None, steps=50,
                               units=None, displacements=None,
                               force_unit='kcal mol$^{-1}$/a$_0$-ish',
                               mass_weight=True,
                               use_internals=False,
                               **opts):
        x, exp_r, exp_t = self.get_distortion_energies(mode, disp_min=disp_min, disp_max=disp_max, steps=steps, order=1,
                                                       mass_weight=mass_weight, use_internals=use_internals,
                                                       displacements=displacements)
        return self.plot_force_comp(mode, x, exp_r, exp_t,
                                    units=units,
                                    force_unit=force_unit,
                                    **opts)

    @classmethod
    def _extrap_solve_insertion(cls, forces, force_r):
        dr_pos = np.searchsorted(force_r, forces)
        drs = []
        for f, d in zip(forces, dr_pos):
            if d == len(force_r):  # linear extrapolation
                c = force_r[-1] - force_r[-2]
                s = (f - force_r[-1]) / c
                drs.append((d, s))
            elif d == 0:  # linear extrapolation
                c = force_r[1] - force_r[0]
                s = (f - force_r[0]) / c
                drs.append((d, s))
            else:
                f0 = force_r[d]
                f1 = force_r[d - 1]
                s = (f0 - f) / (f0 - f1)
                drs.append((d, s))
        return drs

    @classmethod
    def _extrap_solve_energy(cls, x, energies, displacement_pairs, nearest=False):
        es = []
        xs = []
        for start, offset in displacement_pairs:
            if nearest:
                sx = 0
                s = 0
                if start == len(energies):
                    x0 = x[-1]
                    e0 = energies[-1]
                elif start == 0:
                    x0 = x[0]
                    e0 = energies[0]
                else:
                    if offset > .5:
                        x0 = x[start]
                        e0 = energies[start]
                    else:
                        x0 = x[start - 1]
                        e0 = energies[start-1]
            else:
                if start == len(energies):
                    x0 = x[-1]
                    x1 = x[-2]
                    sx = (x0 - x1)
                    e0 = energies[-1]
                    e1 = energies[-2]
                    s = (e0 - e1) #/ sx
                elif start == 0:
                    x0 = x[1]
                    x1 = x[0]
                    sx = (x0 - x1)
                    e0 = energies[1]
                    e1 = energies[0]
                    s = (e0 - e1) #/ sx # offset is negative
                else:
                    x0 = x[start]
                    x1 = x[start - 1]
                    sx = (x0 - x1)
                    e0 = energies[start]
                    e1 = energies[start - 1]
                    s = (e0 - e1) #/ sx
            xs.append(x0 + offset * sx)
            es.append(e0 + offset * s)
        return np.array(xs), np.array(es)

    @classmethod
    def predicted_deltas_from_expansion_energies(cls,
                                                 mode,
                                                 forces,
                                                 x, exp_r, exp_t,
                                                 units=None,
                                                 energy_units='Kilocalories/Mole',
                                                 nearest=False
                                                 ):

        smol = nput.is_numeric(forces)
        if smol: forces = [forces]
        forces = np.asanyarray(forces)
        if units is not None:
            if isinstance(units, str):
                units = units.split("/")
            conv = UnitsData.convert(units[0], "Hartrees") / (
                UnitsData.convert(units[1], "BohrRadius")
            )
            forces = conv * forces

        force_r = exp_r[1][:, mode]
        extrap_pos_r = cls._extrap_solve_insertion(forces, force_r)
        force_t = exp_t[1][:, mode]
        extrap_pos_t = cls._extrap_solve_insertion(forces, force_t)

        conv = UnitsData.convert("Hartrees", energy_units)
        x_r, eng_r = cls._extrap_solve_energy(x, exp_r[0] - np.min(exp_r[0]), extrap_pos_r, nearest=nearest)
        x_t, eng_t = cls._extrap_solve_energy(x, exp_t[0] - np.min(exp_t[0]), extrap_pos_t, nearest=nearest)

        eng_r = conv * eng_r
        eng_t = conv * eng_t

        if smol:
            x_r = x_r[0]
            extrap_pos_r = extrap_pos_r[0]
            eng_r = eng_r[0]
            x_t = x_t[0]
            extrap_pos_t = extrap_pos_t[0]
            eng_t = eng_t[0]

        return (eng_r, x_r, extrap_pos_r), (eng_t, x_t, extrap_pos_t)

    def predicted_delta_from_forces(self,
                                    mode,
                                    forces,
                                    units=None,
                                    disp_min=None,
                                    disp_max=None,
                                    displacements=None,
                                    steps=50,
                                    nearest=False,
                                    mass_weight=True,
                                    max_recursion=5,
                                    max_disp_mag=5000,
                                    use_internals=False,
                                    prev_expansion=None,
                                    ts_initial_coords=None,
                                    rs_initial_coords=None,
                                    return_geometries=False):
        if disp_min is None or nput.is_numeric(disp_min):
            (x, exp_r, exp_t), g = self.get_distortion_energies(mode,
                                                                disp_min=disp_min, disp_max=disp_max, steps=steps,
                                                                order=1,
                                                                mass_weight=mass_weight, use_internals=use_internals,
                                                                displacements=displacements,
                                                                ts_initial_coords=ts_initial_coords,
                                                                rs_initial_coords=rs_initial_coords,
                                                                return_geometries=True)
        else:
            d1, D1 = disp_min
            d2, D2 = disp_max

            if d1 is not None:
                (x1, exp_r1, exp_t1), g1 = self.get_distortion_energies(mode,
                                                                        disp_min=d1, disp_max=D1, steps=steps, order=1,
                                                                        mass_weight=mass_weight,
                                                                        use_internals=use_internals,
                                                                        displacements=displacements,
                                                                        ts_initial_coords=ts_initial_coords,
                                                                        rs_initial_coords=rs_initial_coords,
                                                                        return_geometries=True)
            else:
                x1 = None
            if d2 is not None:
                (x2, exp_r2, exp_t2), g2 = self.get_distortion_energies(mode,
                                                                        disp_min=d2, disp_max=D2, steps=steps, order=1,
                                                                        mass_weight=mass_weight,
                                                                        use_internals=use_internals,
                                                                        displacements=displacements,
                                                                        ts_initial_coords=ts_initial_coords,
                                                                        rs_initial_coords=rs_initial_coords,
                                                                        return_geometries=True)
            else:
                x2 = None

            x_bits = []
            g_r_bits = []
            g_t_bits = []
            e_r_bits = []
            e_t_bits = []

            if x1 is not None:
                if prev_expansion is not None:
                    x1 = x1[:-1]
                    if return_geometries:
                        g1 = (g1[0][:-1], g1[1][:-1])
                x_bits.append(x1)
                if return_geometries:
                    g_r_bits.append(g1[0])
                    g_t_bits.append(g1[1])
                if prev_expansion is not None: # shift for continuitiy
                    exp_r1 = list(exp_r1)
                    exp_r1[0] +=  prev_expansion[1][0][0] - exp_r1[0][-1]
                    exp_r1 = [e[:-1] for e in exp_r1]
                e_r_bits.append(exp_r1)
                if prev_expansion is not None:
                    exp_t1 = list(exp_t1)
                    exp_t1[0] +=  prev_expansion[2][0][0] - exp_t1[0][-1]
                    exp_t1 = [e[:-1] for e in exp_t1]
                e_t_bits.append(exp_t1)
            if prev_expansion is not None:
                if return_geometries:
                    x_bits.append(prev_expansion[0][0])
                    g_r_bits.append(prev_expansion[0][1][0])
                    g_t_bits.append(prev_expansion[0][1][1])
                else:
                    x_bits.append(prev_expansion[0])
                e_r_bits.append(prev_expansion[1])
                e_t_bits.append(prev_expansion[2])
            if x2 is not None:
                if prev_expansion is not None:
                    x2 = x2[1:]
                    if return_geometries:
                        g2 = (g2[0][1:], g2[1][1:])
                x_bits.append(x2)
                if return_geometries:
                    g_r_bits.append(g2[0])
                    g_t_bits.append(g2[1])
                if prev_expansion is not None: # shift for continuitiy
                    exp_r2 = list(exp_r2)
                    exp_r2[0] +=  prev_expansion[1][0][-1] - exp_r2[0][0]
                    exp_r2 = [e[1:] for e in exp_r2]
                e_r_bits.append(exp_r2)
                if prev_expansion is not None: # shift for continuitiy
                    exp_t2 = list(exp_t2)
                    exp_t2[0] +=  prev_expansion[2][0][-1] - exp_t2[0][0]
                    exp_t2 = [e[1:] for e in exp_t2]
                e_t_bits.append(exp_t2)

            x = np.concatenate(x_bits, axis=0)
            if return_geometries:
                g = (
                    np.concatenate(g_r_bits, axis=0),
                    np.concatenate(g_t_bits, axis=0)
                )
            else:
                g = None
            exp_r = [
                np.concatenate(b, axis=0)
                for b in zip(*e_r_bits)
            ]
            exp_t = [
                np.concatenate(b, axis=0)
                for b in zip(*e_t_bits)
            ]

        r_data, ts_data = self.predicted_deltas_from_expansion_energies(
            mode,
            forces,
            x, exp_r, exp_t,
            units=units,
            nearest=nearest
        )
        extrap_pos_r = r_data[1]
        extrap_pos_t = ts_data[1]
        d1 = None
        D1 = None
        d2 = None
        D2 = None
        if (extrap_pos_r < x[0] or extrap_pos_t < x[0]):
            d1 = np.min([extrap_pos_r, extrap_pos_t])
            D1 = x[0]
            d1 = (d1 - D1) * 1.2 + D1

        if (extrap_pos_r > x[-1] or extrap_pos_t > x[-1]):
            d2 = x[-1]
            D2 = np.max([extrap_pos_r, extrap_pos_t])
            D2 = (D2 - d2) * 1.2 + d2

        if return_geometries:
            x = (x, g)

        if (d1 is not None or d2 is not None) and max_recursion > 0:
            return self.predicted_delta_from_forces(
                mode,
                forces,
                units=units,
                disp_min=[d1, D1],
                disp_max=[d2, D2],
                steps=steps,
                nearest=nearest,
                mass_weight=mass_weight,
                max_recursion=max_recursion-1,
                max_disp_mag=max_disp_mag,
                prev_expansion=[x, exp_r, exp_t],
                use_internals=use_internals,
                displacements=displacements,
                return_geometries=return_geometries
            )

        if return_geometries:
            (g_idx_r, g_perc_r) = r_data[2]
            gl_r = g[0]
            if g_perc_r >= 0:
                g_r = gl_r[g_idx_r] * (1-g_perc_r) + gl_r[g_idx_r+1] * g_perc_r
            elif g_idx_r+1 == len(gl_r):
                g_r = gl_r[g_idx_r] + (gl_r[g_idx_r] - gl_r[g_idx_r-1]) * g_perc_r
            elif g_idx_r >= len(gl_r):
                g_r = gl_r[g_idx_r-1] + (gl_r[g_idx_r-1] - gl_r[g_idx_r-2]) * g_perc_r
            else:
                g_r = gl_r[g_idx_r] + (gl_r[g_idx_r+1] - gl_r[g_idx_r]) * g_perc_r # negative

            (g_idx_t, g_perc_t) = ts_data[2]
            gl_t = g[1]
            if g_perc_t >= 0:
                g_t = gl_t[g_idx_t] * (1-g_perc_t) + gl_t[g_idx_t+1] * g_perc_t
            elif g_idx_t+1 == len(gl_t):
                g_t = gl_t[g_idx_t] + (gl_t[g_idx_t] - gl_t[g_idx_t-1]) * g_perc_t
            elif g_idx_t >= len(gl_t):
                g_t = gl_t[g_idx_t-1] + (gl_t[g_idx_t-1] - gl_t[g_idx_t-2]) * g_perc_t
            else:
                g_t = gl_t[g_idx_t] + (gl_t[g_idx_t+1] - gl_t[g_idx_t]) * g_perc_t # negative

            r_data = (r_data[0], (r_data[1], g_r), r_data[2])
            ts_data = (ts_data[0], (ts_data[1], g_t), ts_data[2])

        return ((ts_data[0] - r_data[0]), (r_data, ts_data)) + ((x, exp_r, exp_t),)

    def predicted_internal_delta_from_forces(self,
                                             which,
                                             forces,
                                             displacements=None,
                                             use_internals=True,
                                             **opts
                                             ):
        if displacements is None:
            displacements = self.pure_internal_displacement_matrix
        if not nput.is_int(which):
            which = coordops.zmatrix_indices(self.internals, which)
        return self.predicted_delta_from_forces(
            which,
            forces,
            displacements=displacements,
            use_internals=use_internals,
            **opts
        )


    def predicted_pressure_delta(self,
                                 magnitude,
                                 pressure_units="Megapascals",
                                 *,
                                 pressure=None,
                                 which=0,
                                 pressure_model='xhcff',
                                 pressure_options=None,
                                 # apply_constraints=False,
                                 # remove_orientation=False,
                                 # remove_transrot=False,
                                 displacements=None,
                                 **etc
                                 ):
        if pressure is not None:
            if magnitude is not None:
                raise ValueError("can't get both `magnitude` and `pressure` keywords (they are synonyms)")
            magnitude = pressure
        if isinstance(pressure_units, str):
            pressure_units = pressure_units.replace("pascals", "Pascals").replace("Pascals", "Newtons/MetersSquared")
            pressure_units = pressure_units.rsplit("/", 1)
        force_units, area_units = pressure_units
        area_scaling = UnitsData.convert(area_units, "BohrRadiusSquared")
        if isinstance(force_units, str):
            force_units = force_units.replace("Newtons", "Joules/Meters")
            force_units = force_units.split("/")
        # all of this is to make unit debugging a little easier, did it actually? I don't know
        conv = UnitsData.convert(force_units[0], "Hartrees") / (
            UnitsData.convert(force_units[1], "BohrRadius")
        )
        pressure_scaling = conv / area_scaling
        force_units = "Hartrees/BohrRadius"
        if displacements is None:
            if pressure_options is None:
                pressure_options = {}
            displacements = self.pressure_model_generator(pressure_model,
                                                          pressure=pressure_scaling,
                                                          **pressure_options
                                                          )
        else:
            magnitude = np.asanyarray(magnitude) * pressure_scaling
        return self.predicted_delta_from_forces(
            which,
            magnitude,
            displacements=displacements,
            units=force_units,
            **etc
        )

    def plot_sterics(self,
                     sterics_output,
                     display_cutoff=.5,
                     colormap='coolwarm',
                     stress_sphere_radius=None,
                     stress_sphere_styles=None,
                     stress_point_size=10,
                     plot_ts=True,
                     plot_rs=True,
                     **plot_opts
                     ):
        if stress_sphere_styles is None:
            stress_sphere_styles = {}
        (s_r, s_t), (v, x_r, x_t) = sterics_output
        # (s_r, s_t), (v, x_r, x_t) = fopt.get_distortion_steric_repulsions(
        #     'dihedrals',
        #     max_internals=1,
        #     generate_displacement_function=fopt.generate_internal_distortions,
        #     disp_min=-1,
        #     disp_max=0,
        #     density=2,
        #     return_breakdowns=True
        # )
        pts_r = [np.concatenate(p) for p in s_r[0]]
        vals_r = [np.concatenate(v) for v in s_r[1]]
        pts_t = [np.concatenate(p) for p in s_t[0]]
        vals_t = [np.concatenate(v) for v in s_t[1]]

        min_max = np.min(np.concatenate([
            np.concatenate(vals_r),
            np.concatenate(vals_t)
        ])), np.max(np.concatenate([
                    np.concatenate(vals_r),
                    np.concatenate(vals_t)
                ]))

        keep_pos_r = np.any(
            np.abs(
                nput.vec_rescale(
                    np.moveaxis(np.array(vals_r), 0, -1),
                    [-1, 1],
                    min_max
                )
            ) > display_cutoff,
            axis=-1
        )
        keep_pos_t = np.any(
            np.abs(
                nput.vec_rescale(
                    np.moveaxis(np.array(vals_t), 0, -1),
                    [-1, 1],
                    min_max
                )
            ) > display_cutoff,
            axis=-1
        )
        pts_r = [p[keep_pos_r] for p in pts_r]
        vals_r = [v[keep_pos_r] for v in vals_r]
        pts_t = [p[keep_pos_t] for p in pts_t]
        vals_t = [v[keep_pos_t] for v in vals_t]

        # min_max = np.min(np.concatenate([
        #     np.concatenate(vals_r),
        #     np.concatenate(vals_t)
        # ])), np.max(np.concatenate([
        #             np.concatenate(vals_r),
        #             np.concatenate(vals_t)
        #         ]))


        colors_r = [
            plt.prep_color(palette=colormap, blending=nput.vec_rescale(
                v,
                [0, 1],
                min_max
            ))
            for v in vals_r
        ]

        colors_t = [
            plt.prep_color(palette=colormap, blending=nput.vec_rescale(
                v,
                [0, 1],
                min_max
            ))
            for v in vals_t
        ]

        bits = []
        if plot_ts:
            bits.append(
                self.ts.plot(x_t,
                             annotation_function=lambda mol, i, geom: [
                                 plt.Sphere(p * UnitsData.bohr_to_angstroms, stress_sphere_radius, color=c,
                                            **stress_sphere_styles)
                                 for p, c in zip(pts_t[i], colors_t[i])
                             ] if stress_sphere_radius is not None else [
                                 plt.Point(pts_t[i] * UnitsData.bohr_to_angstroms,
                                           vertex_colors=colors_t[i],
                                           point_size=stress_point_size,
                                           **stress_sphere_styles)
                             ],
                             **plot_opts
                             )
            )
        if plot_rs:
            bits.append(self.rs.plot(x_r,
                          annotation_function=lambda mol, i, geom: [
                              plt.Sphere(p * UnitsData.bohr_to_angstroms, stress_sphere_radius, color=c, **stress_sphere_styles)
                              for p,c in zip(pts_r[i], colors_r[i])
                          ] if stress_sphere_radius is not None else [
                              plt.Point(pts_r[i] * UnitsData.bohr_to_angstroms,
                                         vertex_colors=colors_r[i],
                                        point_size=stress_point_size,
                                         **stress_sphere_styles)
                          ],
                          **plot_opts
                          ))
        if len(bits) == 1:
            return bits[0]
        else:
            return interactive.Grid([
                [b.to_widget() for b in bits]
            ], dynamic=False)#.to_widget().display()