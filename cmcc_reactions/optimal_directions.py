import numpy as np
from McUtils.Data import UnitsData
import McUtils.Numputils as nput
from Psience.Molecools import Molecule

__all__ = [
    "find_optimal_displacement_coordinate",
    "construct_force_dirs",
    "reaction_force_dirs"
]

def clip_f(f):
    f = np.clip(f, -1e15, 1e15)
    if np.abs(f) < 1e-16:
        s = np.sign(f)
        if s == 0: s = 1
        f = s * 1e-15
    return f

def gamma(hess_gs, hess_ts, d):
    f_g = np.dot(np.dot(hess_gs, d), d)
    f_t = np.dot(np.dot(hess_ts, d), d)
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


def find_optimal_displacement_coordinate(gs_hess, ts_hess, proj_dirs,
                                         guess_dir=None,
                                         max_iterations=1000
                                         ):
    if guess_dir is None:
        proj = nput.orthogonal_projection_matrix(proj_dirs)
        f_proj_r = proj @ gs_hess @ proj
        f_proj_ts = proj @ ts_hess @ proj
        guess_dir = get_guess_dir(f_proj_r, f_proj_ts)

    if max_iterations < 0:
        return guess_dir, -1

    def fun(guess, mask):
        return -np.array([gamma(gs_hess, ts_hess, guess[0])])

    def jac(guess, mask):
        gg = dgamma(gs_hess, ts_hess, guess[0])
        # print("!", np.isnan(gg).any())
        return -gg[np.newaxis]

    def fhess(guess, mask):
        return -dgamma2(gs_hess, ts_hess, guess[0])[np.newaxis]

    force_dir, is_opt, error = nput.iterative_step_minimize(
        guess_dir,
        nput.ConjugateGradientStepFinder(fun, jac,
                                         damping_parameter=.9,
                                         restart_interval=20
                                         ),
        # nput.QuasiNewtonStepFinder(fun, jac),#, damping_parameter=.9, restart_interval=10),
        # nput.NewtonStepFinder(fun, jac, fhess,
        #                       line_search=True,
        #                       damping_parameter=.9
        #                      ),
        unitary=True,
        orthogonal_directions=proj_dirs,
        # generate_rotation=True,
        max_iterations=max_iterations
    )

    return force_dir, error


def get_force_dirs(hess_gs, hess_ts, initial_dir, k, **opts):
    initial_dir = np.asanyarray(initial_dir)
    if initial_dir.ndim == 1:
        initial_dir = initial_dir[:, np.newaxis]
    proj_dirs = initial_dir
    errors = []
    for i in range(k):
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
                         max_iterations=1500,
                         **opts
                         ):
    nat = len(modes_gs.masses)
    hess_ts_nms = nm_hess(modes_ts, L=np.eye(modes_ts.matrix.shape[-1]))
    hess_gs_nms = nm_hess(modes_gs, L=modes_ts.matrix.T @ modes_gs.matrix)

    force_dirs, errors = get_force_dirs(hess_gs_nms, hess_ts_nms,
                                        np.eye(nat * 3 - 6)[:, :idx_start],
                                        # new_modes_ts.matrix[:, (0,)],
                                        num_dirs,
                                        max_iterations=max_iterations,
                                        **opts
                                        )
    # gi12 = nput.fractional_power(reactant.get_gmatrix(use_internals=False), -1/2)
    cart_force_dirs = modes_ts.matrix @ force_dirs
    selected_force_dirs = cart_force_dirs.T[idx_start:]

    return selected_force_dirs

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
                        low_frequency_cutoff=None
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
        idx_start = 0

    return construct_force_dirs(new_modes_gs, new_modes_ts,
                                num_dirs=num_dirs,
                                idx_start=idx_start
                                )

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-reactant', help='npz file containing reactant', required=True)
#     parser.add_argument('-ts', help='npz file containing TS', required=True)
#     parser.add_argument('--num_dirs',
#                         help='number of orthogonal force dirs to return',
#                         type=int,
#                         default=12,
#                         )
#     parser.add_argument('--low_freq',
#                         help='low frequency mode cutoff',
#                         type=float,
#                         default=100)
#     parser.add_argument('--frag', help='whether or not modes are fragment localized', type=int, default=-1)
#     args = parser.parse_args()
#
#     reactant = setup_mol(np.load(args.reactant))
#     transition_state = setup_mol(np.load(args.ts))
#
#     new_modes_ts = transition_state.get_normal_modes()
#     new_modes_gs = reactant.get_normal_modes()
#
#     if args.frag >= 0:
#         frag_inds = reactant.fragment_indices[args.frag]
#         new_modes_gs = new_modes_gs.localize(atoms=frag_inds, allow_mode_mixing=True)
#         new_modes_ts = new_modes_ts.localize(atoms=frag_inds, allow_mode_mixing=True)
#
#     lf = args.low_freq
#     ts_freqs = new_modes_ts.freqs * 219474.63
#     idx_start = int(np.where(ts_freqs >= lf)[0][0])
#
#     dirs = construct_force_dirs(new_modes_gs, new_modes_ts,
#                                 num_dirs=args.num_dirs,
#                                 idx_start=idx_start
#                                 )
#
#     write_force_dirs(dirs)
