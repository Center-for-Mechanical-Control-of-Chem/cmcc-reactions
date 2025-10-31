import numpy as np
from scipy.optimize import minimize as scipy_opt
from McUtils.Data import UnitsData
from McUtils.Scaffolding import Logger
import McUtils.Devutils as dev
import McUtils.Numputils as nput
from Psience.Molecools import Molecule
from Psience.Modes import MixtureModes

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
                         max_iterations=None,
                         **opts
                         ):
    # nat = len(modes_gs.masses)
    hess_ts_nms = nm_hess(modes_ts, L=np.eye(modes_ts.matrix.shape[-1]))
    hess_gs_nms = nm_hess(modes_gs, L=modes_ts.inverse @ modes_gs.matrix)

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

def reaction_force_dirs(reactant, transition_state,
                        num_dirs=6,
                        fragment_indices=None,
                        low_frequency_cutoff=0.00045, # 100 cm-1
                        return_modes=True,
                        **opts
                        ):
    new_modes_ts = transition_state.get_normal_modes()
    new_modes_gs = reactant.get_normal_modes()

    if fragment_indices is not None:
        if isinstance(fragment_indices, int):
            fragment_indices = reactant.fragment_indices[fragment_indices]
        new_modes_gs = new_modes_gs.localize(atoms=fragment_indices, allow_mode_mixing=True)
        new_modes_ts = new_modes_ts.localize(atoms=fragment_indices, allow_mode_mixing=True)

    if low_frequency_cutoff is not None:
        ts_freqs = new_modes_ts.freqs
        idx_start = int(np.where(ts_freqs >= low_frequency_cutoff)[0][0])
    else:
        idx_start = 1

    dirs = construct_force_dirs(new_modes_gs, new_modes_ts,
                                num_dirs=num_dirs,
                                idx_start=idx_start,
                                **opts
                                )
    if return_modes:
        return dirs, (new_modes_gs, new_modes_ts)
    else:
        return dirs

def compute_reaction_gamma(reactant, transition_state, direction_gs,
                           direction_ts=None,
                           use_mode_space=True
                           ):
    direction_gs = np.asanyarray(direction_gs)
    if direction_ts is None:
        direction_ts = direction_gs
    else:
        direction_ts = np.asanyarray(direction_ts)
    new_modes_ts = transition_state.get_normal_modes()
    new_modes_gs = reactant.get_normal_modes()

    if use_mode_space:
        f_ts = nm_hess(new_modes_ts, L=np.eye(new_modes_ts.matrix.shape[-1]))
        f_gs = nm_hess(new_modes_gs, L=new_modes_ts.inverse @ new_modes_gs.matrix)
        direction_gs = np.dot(direction_gs, new_modes_ts.modes_by_coords)
        direction_ts = np.dot(direction_ts, new_modes_ts.modes_by_coords)
    else:
        f_gs = new_modes_gs.compute_hessian('coords')
        f_ts = new_modes_ts.compute_hessian('coords')

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

def reorder_force_dirs(rs_solv, ts_solv, dirs, direction_ts=None, use_mode_space=True):
    gamma_list = compute_reaction_gamma(rs_solv, ts_solv, dirs, direction_ts=direction_ts, use_mode_space=use_mode_space)
    ord_g = np.argsort(-gamma_list)
    if direction_ts is not None:
        dirs = (dirs[ord_g,], direction_ts[ord_g,])
    else:
        dirs = dirs[ord_g,]
    return gamma_list[ord_g,], dirs

def mass_weighted_normalize_displacements(mol, expansion=None):
    if expansion is None:
        expansion = mol.get_cartesians_by_internals(1)[0]
    else:
        expansion = np.asanyarray(expansion)
        if expansion.ndim == 3:
            expansion = expansion[0]
    return nput.vec_normalize(
        expansion @ mol.get_gmatrix(power=-1/2, use_internals=False),
        axis=1
    ) @ mol.get_gmatrix(power=1/2, use_internals=False)

class ForceOptimizer:
    default_options = {
        'num_dirs':50,
        'optimizer':'scipy',
        'method':'cg',
        'max_iterations':100
    }
    def __init__(self, reactant_mol, ts_mol, optimal_forces=None, reorder=True, **determination_opts):
        self.rs = reactant_mol
        self.ts = ts_mol
        self.opts = dict(self.default_options, **determination_opts)
        self.reorder = reorder
        self._optimal_forces = optimal_forces

    @classmethod
    def from_displacements(cls, reactant_mol, ts_mol, dirs_gs, dirs_ts=None, reorder=False, modes=None):
        if modes is None:
            modes = (None, None)
        rs_modes, ts_modes = modes
        if rs_modes is None:
            if reactant_mol.potential_derivatives is None:
                reactant_mol.potential_derivatives = reactant_mol.calculate_energy(order=2)[1:]
            rs_modes = reactant_mol.get_normal_modes()
        if ts_modes is None:
            if ts_mol.potential_derivatives is None:
                ts_mol.potential_derivatives = ts_mol.calculate_energy(order=2)[1:]
            ts_modes = ts_mol.get_normal_modes()

        dirs_gs = mass_weighted_normalize_displacements(reactant_mol, dirs_gs)
        if dirs_ts is not None:
            dirs_ts = mass_weighted_normalize_displacements(reactant_mol, dirs_ts)

        if reorder:
            gammas, dirs = reorder_force_dirs(reactant_mol, ts_mol, dirs_gs, dirs_ts, use_mode_space=True)
        else:
            gammas = compute_reaction_gamma(reactant_mol, ts_mol, dirs_gs, dirs_ts, use_mode_space=True)
            if dirs_ts is not None:
                dirs = (dirs_gs, dirs_ts)
            else:
                dirs = dirs_gs
        return cls(reactant_mol, ts_mol, optimal_forces=((gammas, dirs), (rs_modes, ts_modes)))

    @classmethod
    def from_internals(cls, reactant_mol, ts_mol, internal_spec, active_atoms=None, fixed_atoms=None, **opts):
        if fixed_atoms is None and active_atoms is not None:
            fixed_atoms = np.setdiff1d(np.arange(len(reactant_mol.coords)), active_atoms)

        # rs_dist.fragment_indices[1][(4, 5, 6, 7),]
        _, disp_gs = nput.internal_coordinate_tensors(
            reactant_mol.coords,
            internal_spec,
            fixed_atoms=fixed_atoms,
            masses=reactant_mol.atomic_masses,
            return_inverse=True
        )
        _, disp_ts = nput.internal_coordinate_tensors(
            ts_mol.coords,
            internal_spec,
            fixed_atoms=fixed_atoms,
            masses=ts_mol.atomic_masses,
            return_inverse=True
        )

        return cls.from_displacements(reactant_mol, ts_mol, disp_gs, disp_ts, **opts)

    def optimize(self):
        if self.rs.potential_derivatives is None:
            self.rs.potential_derivatives = self.rs.calculate_energy(order=2)[1:]
        if self.ts.potential_derivatives is None:
            self.ts.potential_derivatives = self.ts.calculate_energy(order=2)[1:]

        (dirs, _), modes = reaction_force_dirs(self.rs, self.ts, **self.opts)
        if self.reorder:
            gammas, dirs = reorder_force_dirs(self.rs, self.ts, dirs, use_mode_space=True)
        else:
            gammas = compute_reaction_gamma(self.rs, self.ts, dirs, use_mode_space=True)

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
    def rs_modes(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        return self._optimal_forces[1][0]
    @property
    def ts_modes(self):
        if self._optimal_forces is None:
            self._optimal_forces = self.optimize()
        return self._optimal_forces[1][1]

    def animate_normed(self, i, expansion=None, use_internals=False, mag=.5, mol='ts'):
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
                                      coordinate_expansion=[nput.vec_normalize(exp, axis=1)]
                                      )



