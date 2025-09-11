
import numpy as np
import os
import collections
import itertools
import pprint
from McUtils.Data import AtomData, UnitsData
import McUtils.Numputils as nput
# import McUtils.Jupyter as interactive
import McUtils.Iterators as itut
from Psience.Molecools import Molecule

import McUtils.Plots as plt
# import McUtils.Devutils as dev
# from Psience.Modes import NormalModes
# import McUtils.Coordinerds as coordops

def conf_path(reaction_class, idx_spec, *subpaths):
    if len(idx_spec) == 3:
        idx_spec = list(idx_spec[:2]) + [f'force_{idx_spec[2]}']
    return os.path.join(reaction_class, *[str(x) for x in idx_spec], *subpaths)
def load_mol(reaction_class, idx_spec, key):
    base_path = conf_path(reaction_class, idx_spec, f'{key}.xyz' if len(idx_spec) == 2 else f'{key}_force.xyz')
    return Molecule.from_file(base_path, units='Angstroms')
def load_reactant(reaction_class, idx_spec):
    return load_mol(reaction_class, idx_spec, 'Reactant')
def load_ts(reaction_class, idx_spec):
    return load_mol(reaction_class, idx_spec, 'TS')
def load_product(reaction_class, idx_spec):
    return load_mol(reaction_class, idx_spec, 'Product')
def load_force(reaction_class, idx_spec):
    return np.loadtxt(conf_path(reaction_class, idx_spec, 'constraints.txt')) * (
        UnitsData.convert("ElectronVolts", "Kilocalories/Mole") /
        UnitsData.convert("Angstroms", "BohrRadius")
    )

def reaction_index_iter(reaction_class, return_conf=False, include_toplevel=False):
    for o in os.listdir(reaction_class):
        targ = os.path.join(reaction_class, o)
        if not os.path.isdir(targ): continue
        try:
            idx = int(o)
        except:
            continue
        for j in os.listdir(targ):
            targ = os.path.join(reaction_class, o, j)
            if not os.path.isdir(targ): continue
            try:
                idx = int(j)
            except:
                continue
            if return_conf:
                yield (o, j)
            else:
                if include_toplevel:
                    yield (o, j)
                for k in os.listdir(targ):
                    if k.startswith('force_'):
                        yield (o, j, k[len('force_'):])


ReactionProfileData = collections.namedtuple('ReactionProfileData', ['energies', 'geometries', 'absolute_energies'])
def parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=False):
    conv = conf_path(reaction_class, idx_spec, 'opt_converged_000.xyz')
    if not os.path.isfile(conv): return None
    with open(conv) as rp_data:
        rp_string = rp_data.read()

        chunks, eng_block = rp_string.split('[GEOCONV', 1)
        chunks = chunks.split('(XYZ)')[1].split("\n\n")
        geoms = []
        for c in chunks[1:]:
            xyz = c.rsplit("\n", 1)[0]
            geoms.append(
                Molecule.from_string(xyz, 'xyz', units='Angstroms', energy_evaluator='aimnet2')
            )

        eng_block = eng_block.split("energy", 1)[1].split("max-force", 1)[0].strip()
        engs = np.array(eng_block.splitlines()).astype('float')
        if load_absolute_energies:
            abs_eng = get_absolute_energies(geoms)
        else:
            abs_eng = None
        return ReactionProfileData(engs, geoms, abs_eng)


def get_absolute_energies(geometries):
    return geometries[0].calculate_energy(
        np.array([g.coords for g in geometries])
    ) * UnitsData.convert("Hartrees", "Kilocalories/Mole")


BarrierData = collections.namedtuple('BarrierData', ['conf_inds', 'force', 'no_force', 'delta'])
def load_barrier_data(reaction_class):
    conf_inds = np.load(os.path.join(reaction_class, 'conf_indices.npy'))
    force_barriers = np.load(os.path.join(reaction_class, 'Energy_force.npy'))[:, 0] * UnitsData.convert(
        "ElectronVolts", "Kilocalories/Mole")
    barriers = np.load(os.path.join(reaction_class, 'Energy_no_force.npy'))[:, 0] * UnitsData.convert("ElectronVolts",
                                                                                                      "Kilocalories/Mole")
    deltas = force_barriers - barriers
    return BarrierData(conf_inds, force_barriers, barriers, deltas)


ReactionData = collections.namedtuple('ReactionData', ['reactant', 'ts', 'product', 'force'])


def load_reaction_data(reaction_class, idx_spec):
    return ReactionData(
        load_reactant(reaction_class, idx_spec),
        load_ts(reaction_class, idx_spec),
        load_product(reaction_class, idx_spec),
        load_force(reaction_class, idx_spec)
    )


ForceTolData = collections.namedtuple('ForceTolData', ['indices', 'force_magnitudes'])
def load_force_tolerances(reaction_class):
    f_rea = np.load(os.path.join(reaction_class, 'F_rea.npy'))
    conf_inds = np.load(os.path.join(reaction_class, 'F_rea_conf_indices.npy'))

    return ForceTolData(conf_inds, f_rea)

def animate_reaction(reaction_class, idx_spec):
    geoms = parse_reaction_path(reaction_class, idx_spec)
    anim_name = "_".join(['anim', reaction_class] + list(idx_spec))
    return geoms[0].plot(
        geoms[0].embed_coords([g.coords for g in geoms]),
        animation_options=dict(animation_duration=5),
        include_save_buttons=True,
        image_size=800
    ).to_widget().write(anim_name + '.html')

def plot_forces(reaction_class, idx_spec, structure_index=0, scaling=10, **opts):
    geom_test = parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=False)
    reaction_test = load_reaction_data(reaction_class, idx_spec)

    return geom_test.geometries[structure_index].plot(
        backend='x3d',
        mode_vectors=reaction_test.force * scaling,
        **opts
    )

def get_diene_embedding(geom, c1, c2):
    check_bonds = set(tuple(sorted(b[:2])) for b in geom.bonds)
    for i,j in check_bonds:
        if j == c1:
            i,j = j,i
        if i == c1:
            for k,l in check_bonds:
                if l == j:
                    k,l = l,k
                if k == j:
                    for x, y in check_bonds:
                        if l == y:
                            x, y = y,x
                        if x == l and y == c2:
                            return [c1, j, l, c2]

def compare_geometries(reaction_class, idx_spec, embedding_atoms=[4, 11], structure_index=0, **opts):
    geom_test = parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=False)
    geom_test_nf = parse_reaction_path(reaction_class, idx_spec[:2], load_absolute_energies=False)

    geom = geom_test.geometries[structure_index]
    geom_nf = geom_test_nf.geometries[structure_index]
    if len(embedding_atoms) == 2:
        embedding_atoms = get_diene_embedding(geom, *embedding_atoms)
    geom_f = geom.get_embedded_molecule(load_properties=False)
    geom_nf = geom_nf.get_embedded_molecule(ref=geom_f, load_properties=False, sel=list(sorted(embedding_atoms)))
    g1 = geom_f.plot(backend='x3d', **opts)
    geom_nf.plot(figure=g1, backend='x3d', highlight_atoms=list(range(len(geom_nf.atoms))))
    return g1

def get_pair_rmsd(a, b):
    diff = np.asanyarray(a) - np.asanyarray(b)
    return np.linalg.norm(diff.flatten(), axis=-1) / np.sqrt(len(diff.flatten()))

def get_incremental_RMSD(geometries):
    coords = geometries[0].embed_coords(np.array([g.coords for g in geometries]))
    diffs = np.diff(coords, axis=0)
    rmsds = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=-1) / np.sqrt(3 * len(geometries[0].atoms))
    return nput.vec_rescale(np.cumsum([0] + list(rmsds)))

def get_true_reaction_barrier(reaction_class, idx_spec):
    engs = parse_reaction_path(reaction_class, idx_spec).energies
    ts_pos = np.argmax(engs)
    if ts_pos == 0: return 0
    return engs[ts_pos] - np.min(engs[:ts_pos])

def check_reaction(reaction_class, idx_spec):
    engs = parse_reaction_path(reaction_class, idx_spec)
    if engs is None: return False
    engs = engs.energies
    ts_pos = np.argmax(engs)
    if ts_pos == 0: return False
    return np.min(engs[:ts_pos]) >= engs[0]

def check_reaction_class(reaction_class, top_level=True):
    bad_rxns = []
    for x in itut.delete_duplicates(y[:2] for y in reaction_index_iter(reaction_class, return_conf=top_level)):
        if not check_reaction(reaction_class, x):
            print(x)
            bad_rxns.append(x)
    file = f'{reaction_class}_bad_reactions.txt' if not top_level else f'{reaction_class}_bad_toplevel.txt'
    with open(file, 'w+') as bad:
        print(pprint.pformat(bad_rxns), file=bad)
    return bad_rxns

def plot_reaction_profiles(reaction_class, idx_spec, **opts):
    geom_test = parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=True)
    geom_test_nf = parse_reaction_path(reaction_class, idx_spec[:2], load_absolute_energies=True)
    fig1 = plt.Plot(
        get_incremental_RMSD(geom_test.geometries),
        geom_test.absolute_energies - geom_test_nf.absolute_energies[0],
        **collections.ChainMap(
            dict(
                aspect_ratio=3/4,
                axes_labels=["R", 'Energy (kcal mol$^{-1}$)'],
                label='Force',
                plot_legend=True
            ),
            opts
        )
    )
    plt.Plot(
        get_incremental_RMSD(geom_test_nf.geometries),
        geom_test_nf.absolute_energies - geom_test_nf.absolute_energies[0],
        # aspect_ratio=3/4,
        # axes_labels=["R", 'Energy (kcal mol$^{-1}$)'],
        figure=fig1,
        label='No Force'
    )
    # fig1.savefig("profile_3_2_1.png")
    return fig1

def remove_transrot(mol, mode):
    transrot_modes = mol.translation_rotation_modes[1]
    p = np.eye(transrot_modes.shape[0]) - transrot_modes @ transrot_modes.T
    g12 = mol.get_gmatrix(use_internals=False, power=-1/2) # for mass-weighting
    gi12 = mol.get_gmatrix(use_internals=False, power=1/2)
    p = gi12 @ p @ g12
    proj_vec = p @ np.asanyarray(mode).flatten()
    return proj_vec

def uniform_displacement_vector(nats, sels_to_axes):
    z = np.zeros((nats, 3))
    for s,a in sels_to_axes.items():
        z[s,] = np.asanyarray(a)[np.newaxis]
    return z

def get_lab_frame_force(mol, test_vector, target_force):
    transrot_modes = mol.translation_rotation_modes[1]
    g12 = mol.get_gmatrix(use_internals=False, power=-1/2) # for mass-weighting
    gi12 = mol.get_gmatrix(use_internals=False, power=1/2)
    target_force = g12 @ np.asanyarray(target_force.flatten())[:, np.newaxis]
    test_vector = g12 @ np.asanyarray(test_vector.flatten())[:, np.newaxis]
    basis = np.concatenate([transrot_modes, test_vector], axis=1)
    tf = np.linalg.lstsq(basis, target_force, rcond=None)[0]
    return (gi12 @ (basis @ tf))

def get_random_lab_force(mol, test_vector, n=1):
    transrot_modes = mol.translation_rotation_modes[1]
    g12 = mol.get_gmatrix(use_internals=False, power=-1/2) # for mass-weighting
    gi12 = mol.get_gmatrix(use_internals=False, power=1/2)
    test_vector = nput.vec_normalize(g12 @ np.asanyarray(test_vector.flatten())[:, np.newaxis])
    basis = np.concatenate([transrot_modes, test_vector], axis=1)
    tf = nput.vec_normalize(np.random.uniform(size=(7, n)), axis=0)
    return (gi12 @ (basis @ tf))

def compile_reaction_class(reaction_class):
    inds = []
    atom = []
    geoms = []
    engs = []
    for ind in reaction_index_iter(reaction_class):
        rpd = parse_reaction_path(reaction_class, ind)
        if rpd is not None:
            inds.append([int(x) for x in ind])
            atom.append(rpd.geometries[0].atoms)
            geoms.append(np.concatenate([g.coords for g in rpd.geometries], axis=0))
            engs.append(rpd.energies)

    return {
        'inds':inds,
        'atoms':atom,
        'coords':geoms,
        'energies':engs
    }

def write_aggregate_data(file, aggregate_data):
    remapped_data = {}
    for k,v in aggregate_data.items():
        flat_val = np.concatenate(v, axis=0)
        inds = np.array([len(f) for f in v])
        remapped_data[k] = flat_val
        remapped_data[k+"_ind_map"] = inds

    return np.savez(file, **remapped_data)

def load_aggregate_data(file):
    aggregate_data = np.load(file)
    res = {}
    for k,v in aggregate_data.items():
        if k.endswith('_ind_map'): continue
        blocks = aggregate_data[k+"_ind_map"]
        split_pos = np.cumsum(blocks)[:-1]
        chunks = np.array_split(v, split_pos)
        res[k] = chunks

    return res
