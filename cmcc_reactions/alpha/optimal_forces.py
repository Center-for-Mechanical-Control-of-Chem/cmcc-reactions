from McUtils import Numputils as nput
import numpy as np



# nput = McUtils.load_module('.Numputils')
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
                                         max_iterations=1000,
                                         optimizer='newton',
                                         **optimization_parameters
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


# force_dirs, errors = get_force_dirs(hess_gs_nms, hess_ts_nms,
#                                     np.eye(150)[:, :5],
#                                     # new_modes_ts.matrix[:, (0,)],
#                                     10,
#                                     max_iterations=1500
#                                     )
# g12 = nput.fractional_power(reactant.get_gmatrix(use_internals=False), 1 / 2)
# cart_force_dirs = force_dirs.T @ new_modes_ts.inverse @ g12
# [
#     gamma(hess_gs_nms, hess_ts_nms, force_dirs[:, 1]),
#     errors
# ]