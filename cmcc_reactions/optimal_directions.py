import collections

import numpy as np
from scipy.optimize import minimize as scipy_opt
from McUtils.Data import UnitsData
from McUtils.Scaffolding import Logger
import McUtils.Devutils as dev
import McUtils.Numputils as nput
from Psience.Molecools import Molecule
from Psience.Modes import MixtureModes
import McUtils.Plots as plt

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

def gamma(hess_gs, hess_ts, d, d_ts=None):
    if d_ts is None:
        d_ts = d
    f_g = np.dot(np.dot(hess_gs, d), d)
    f_t = np.dot(np.dot(hess_ts, d_ts), d_ts)
    return 1 / f_g - 1 / f_t

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

def scipy_optimize_forces(gs_hess, ts_hess, guess_dir, proj_dirs, *, max_iterations, logger=None,
                          method='nelder-mead'):
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

    opts = dict(options={'maxiter':max_iterations})
    if method in {'cg', 'bfgs'}:
        opts['jac'] = jac

    if logger is not None:
        logger = Logger.lookup(logger)
        prev_re = [guess_dir]
        opts['callback'] = lambda intermediate_result, prev_re=prev_re: (
            logger.log_print(
                [
                    "Struct: {intermediate_result}",
                    "Step: {intermediate_step}"
                ],
                intermediate_result=intermediate_result,
                intermediate_step=intermediate_result - prev_re[-1]
            ),
            prev_re.append(intermediate_result)
        )

    min = scipy_opt(fun, guess_dir, method=method, **opts)
    return np.dot(nput.vec_normalize(min.x), reduced_basis.T), min

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
        ]
    )
    return reactant


def construct_force_dirs(modes_gs, modes_ts,
                         *,
                         num_dirs,
                         idx_start,
                         mols=None,
                         max_iterations=None,
                         use_mode_space=True,
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
    else:
        hess_gs_nms = modes_gs.compute_hessian('coords')
        hess_ts_nms = modes_ts.compute_hessian('coords')

    # hess_ts_nms = nm_hess(modes_ts, L=np.eye(modes_ts.matrix.shape[-1]))
    # hess_gs_nms = nm_hess(modes_gs, L=modes_ts.inverse @ modes_gs.matrix)

    force_dirs, errors = get_force_dirs(hess_gs_nms, hess_ts_nms,
                                        np.eye(len(hess_ts_nms))[:, :idx_start],
                                        # new_modes_ts.matrix[:, (0,)],
                                        num_dirs,
                                        max_iterations=max_iterations,
                                        **opts
                                        )
    cart_force_dirs = force_dirs.T @ modes_ts.coords_by_modes
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

LOW_FREQUENCY_MODE_CUTOFF = 0.00045
def reaction_force_dirs(reactant, transition_state,
                        num_dirs=6,
                        fragment_indices=None,
                        low_frequency_cutoff=None, # 100 cm-1
                        return_modes=True,
                        extra_localization=None,
                        **opts
                        ):
    new_modes_ts = transition_state.get_normal_modes()
    new_modes_gs = reactant.get_normal_modes()

    if fragment_indices is not None:
        if isinstance(fragment_indices, int):
            fragment_indices = reactant.fragment_indices[fragment_indices]
        new_modes_gs = new_modes_gs.localize(atoms=fragment_indices, allow_mode_mixing=True)
        new_modes_ts = new_modes_ts.localize(atoms=fragment_indices, allow_mode_mixing=True)

    if extra_localization is not None:
        extra_localization = extra_localization | dict(allow_mode_mixing=True)
        new_modes_gs = new_modes_gs.localize(**extra_localization)
        new_modes_ts = new_modes_ts.localize(**extra_localization)

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
                                **opts
                                )
    if return_modes:
        return dirs, (new_modes_gs, new_modes_ts)
    else:
        return dirs

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
                           modes=None
                           ):
    (direction_gs, direction_ts), (f_gs, f_ts) = prep_gamma_hessians(
        reactant, transition_state,
        direction_gs, direction_ts, use_mode_space, modes
    )

    if direction_gs.ndim > 1:
        return np.array([
            gamma(f_gs, f_ts, d, dt)
            for d,dt in zip(direction_gs, direction_ts)
        ])
    else:
        return gamma(
            f_gs,
            f_ts,
            direction_gs,
            direction_ts
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

def mass_weighted_normalize_displacements(mol, expansion=None, inverse=None, orthogonalize=False, mode='forward'):
    if mode == 'forward':
        if expansion is None:
            expansion = mol.get_cartesians_by_internals(1)[0]
        else:
            expansion = np.asanyarray(expansion) #TODO: why?
            if expansion.ndim == 3:
                expansion = expansion[0]
        b = expansion @ mol.get_gmatrix(power=-1/2, use_internals=False)
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
        b = b @ mol.get_gmatrix(power=1/2, use_internals=False)
        if inverse is not None:
            return b, inverse
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
            b = mol.get_gmatrix(power=1/2, use_internals=False) @ inverse
            if orthogonalize:
                b, r = np.linalg.qr(b)
                pinv = np.linalg.inv(expansion @ mol.get_gmatrix(power=-1/2, use_internals=False) @ b)
                expansion = pinv @ expansion
            else:
                b, norms = nput.vec_normalize(b, axis=0, return_norms=True)
                expansion = np.diag(norms) @ expansion
            b = mol.get_gmatrix(power=-1/2, use_internals=False) @ b
            return expansion, b
        else:
            if expansion is None:
                expansion = mol.get_internals_by_cartesians(1)[0]
            else:
                expansion = np.asanyarray(expansion)  # TODO: why?
                if expansion.ndim == 3:
                    expansion = expansion[0]
            b = mol.get_gmatrix(power=1/2, use_internals=False) @ expansion
            if orthogonalize:
                b, r = np.linalg.qr(b)
            else:
                b, norms = nput.vec_normalize(b, axis=0, return_norms=True)
            b = mol.get_gmatrix(power=-1/2, use_internals=False) @ b
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

class ForceOptimizer:
    default_options = {
        'num_dirs':50,
        'optimizer':'scipy',
        'method':'cg',
        'max_iterations':100
    }
    def __init__(self, reactant_mol, ts_mol, optimal_forces=None, reorder=True, inverse_forces=None,
                 use_mode_space=True,
                 reembed=True,
                 **determination_opts):
        if reembed:
            reactant_mol = reactant_mol.get_embedded_molecule(ref=ts_mol)
        self.rs = reactant_mol
        self.ts = ts_mol
        self.opts = dict(self.default_options, **determination_opts)
        self.reorder = reorder
        self._optimal_forces = optimal_forces
        self._inverse = inverse_forces
        self.use_mode_space = use_mode_space

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

    def optimize(self):
        if self.rs.potential_derivatives is None:
            self.rs.potential_derivatives = self.rs.calculate_energy(order=2)[1:]
        if self.ts.potential_derivatives is None:
            self.ts.potential_derivatives = self.ts.calculate_energy(order=2)[1:]

        (dirs, _), modes = reaction_force_dirs(self.rs, self.ts, use_mode_space=self.use_mode_space, **self.opts)
        if self.reorder:
            gammas, dirs = reorder_force_dirs(self.rs, self.ts, dirs, use_mode_space=self.use_mode_space)
        else:
            gammas = compute_reaction_gamma(self.rs, self.ts, dirs, use_mode_space=self.use_mode_space)

        return (gammas, dirs), modes

    @property
    def gammas(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        return self._optimal_forces[0][0]
    @property
    def force_dirs(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        return self._optimal_forces[0][1]
    @property
    def force_dirs_inverse(self):
        if self._inverse is None:
            fds = self._optimal_forces[0][1]
            if isinstance(fds, np.ndarray):
                self._inverse = mass_weighted_displacement_inverse(self.ts, fds)
            else:
                self._inverse = (
                    mass_weighted_displacement_inverse(self.rs, fds[0]),
                    mass_weighted_displacement_inverse(self.ts, fds[1])
                )
        return self._inverse
    @property
    def rs_modes(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        return self._optimal_forces[1][0]
    @property
    def ts_modes(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        return self._optimal_forces[1][1]

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

    def animate_normed(self, i, expansion=None, modes=None, use_internals=False, mag=.5, mol='ts', **opts):
        if expansion is None and not use_internals:
            expansion = self.force_dirs
        if dev.str_is(mol, 'ts'):
            mol = self.ts
            if len(expansion) == 2 and expansion[0].ndim == 2:
                expansion = expansion[1]
        elif dev.str_is(mol, 'reactant'):
            mol = self.rs
            if len(expansion) == 2 and expansion[0].ndim == 2:
                expansion = expansion[0]
        exp = mass_weighted_normalize_displacements(mol, expansion=expansion)
        return mol.animate_coordinate(i, mag,
                                      coordinate_expansion=[nput.vec_normalize(exp, axis=1)],
                                      **opts
                                      )

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

    def get_displaced_geometries(self, mode, disp_min=-50, disp_max=50, steps=50):
        scan_coords_r = self.rs.get_scan_coordinates(
            [[disp_min, disp_max, steps]],
            which=[mode],
            coordinate_expansion=[self.force_dirs]
        )
        scan_coords_t = self.ts.get_scan_coordinates(
            [[disp_min, disp_max, steps]],
            which=[mode],
            coordinate_expansion=[self.force_dirs]
        )

        return np.linspace(disp_min, disp_max, steps), scan_coords_r, scan_coords_t

    def get_distortion_energies(self, mode, disp_min=-50, disp_max=50, steps=50, order=None, shift=True):
        x, sr, st = self.get_displaced_geometries(mode, disp_min=disp_min, disp_max=disp_max, steps=steps)
        eng_r = self.rs.calculate_energy(coords=sr, order=order)
        eng_ts = self.ts.calculate_energy(coords=st, order=order)
        if order is None:
            if shift:
                eng_r, eng_ts = eng_r - np.min(eng_r), eng_ts - np.min(eng_ts)
        else:
            if shift:
                eng_r[0] = eng_r[0] - np.min(eng_r[0])
                eng_ts[0] = eng_ts[0] - np.min(eng_ts[0])
            eng_r = eng_r[:1] + nput.tensor_reexpand([self.force_dirs], eng_r[1:], axes=[-1, -1])
            eng_ts = eng_ts[:1] + nput.tensor_reexpand([self.force_dirs], eng_ts[1:], axes=[-1, -1])

        return x, eng_r, eng_ts

    def get_distortion_forces(self, mode, disp_min=-50, disp_max=50, steps=50):
        x, sr, st = self.get_displaced_geometries(mode, disp_min=disp_min, disp_max=disp_max, steps=steps)
        eng_r = self.rs.calculate_energy(coords=sr)
        eng_ts = self.ts.calculate_energy(coords=st)

        return x, eng_r - np.min(eng_r), eng_ts - np.min(eng_ts)

    @classmethod
    def plot_eng_comp(cls, x, eng_r, eng_ts, **opts):
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

    def plot_distortion_energies(self, mode, disp_min=-50, disp_max=50, steps=50, **opts):
        x, eng_r, eng_ts = self.get_distortion_energies(mode, disp_min=disp_min, disp_max=disp_max, steps=steps)
        return self.plot_eng_comp(x, eng_r, eng_ts, **opts)

    def plot_distortion_forces(self, mode, disp_min=-50, disp_max=50, steps=50,
                               units=None,
                               force_unit='kcal mol$^{-1}$/a$_0$-ish',
                               **opts):
        x, exp_r, exp_t = self.get_distortion_energies(mode, disp_min=disp_min, disp_max=disp_max, steps=steps, order=1)
        e_r = exp_r[1][:, mode]
        e_t = exp_t[1][:, mode]
        if units is not None:
            if isinstance(units, str):
                units = units.split("/")
            conv = UnitsData.convert("Hartrees", units[0]) / (
                UnitsData.convert("BohrRadius", units[1])
            ) * UnitsData.convert("Kilocalories/Mole", "Hartrees")
            e_r = e_r * conv
            e_t = e_t * conv

        return self.plot_eng_comp(x,
                                  e_r,
                                  e_t,
                                  axes_labels=["x ($a_0$-ish)", fr"$\partial E/\partial x$ ({force_unit})"],
                                  **opts
                                  )  # .savefig('/Users/Mark/Desktop/sample_distortion_forces.png')
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
                                    disp_min=-50,
                                    disp_max=50,
                                    steps=50,
                                    nearest=False
                                    ):
        x, exp_r, exp_t = self.get_distortion_energies(mode,
                                                       disp_min=disp_min, disp_max=disp_max, steps=steps, order=1)

        preds = self.predicted_deltas_from_expansion_energies(
            mode,
            forces,
            x, exp_r, exp_t,
            units=units,
            nearest=nearest
        )

        return preds + ((x, exp_r, exp_t),)