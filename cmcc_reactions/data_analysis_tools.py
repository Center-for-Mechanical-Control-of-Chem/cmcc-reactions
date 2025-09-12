
import numpy as np
import os
import collections
import itertools
import pprint
from McUtils.Data import AtomData, UnitsData
import McUtils.Numputils as nput
import McUtils.Parsers as parsers
# import McUtils.Jupyter as interactive
import McUtils.Iterators as itut
from Psience.Molecools import Molecule

import McUtils.Plots as plt
# import McUtils.Devutils as dev
# from Psience.Modes import NormalModes
# import McUtils.Coordinerds as coordops

def conf_path(reaction_class, idx_spec, *subpaths):
    if len(idx_spec) == 3:
        idx_spec = list(idx_spec[:-1]) + [f'force_{idx_spec[2]}']
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
def parse_reaction_trajectory(conv, load_absolute_energies=False,
                              load_mols=True,
                              parse_struct=True):
    with open(conv) as rp_data:
        rp_string = rp_data.read()

        chunks, eng_block = rp_string.split('[GEOCONV', 1)
        chunks = chunks.split('(XYZ)')[1].split("\n\n")
        if parse_struct:
            geoms = []
            for c in chunks[1:]:
                xyz = c.rsplit("\n", 1)[0]
                if load_mols:
                    geoms.append(
                        Molecule.from_string(xyz, 'xyz', units='Angstroms', energy_evaluator='aimnet2')
                    )
                else:
                    geoms.append({
                        'atoms':parsers.Word.findall(xyz),
                        'coords':np.array(parsers.Number.findall(xyz)).astype('float').reshape(-1, 3)
                    })
        else:
            geoms = chunks[1:]

        eng_block, max_force = eng_block.split("energy", 1)[1].split("max-force", 1)
        eng_block = eng_block.strip()
        engs = np.array(eng_block.splitlines()).astype('float')
        if load_absolute_energies:
            abs_eng = get_absolute_energies(geoms)
        else:
            abs_eng = None
        return ReactionProfileData(engs, geoms, abs_eng)

def parse_reaction_trajectory_forces(conv):
    with open(conv) as rp_data:
        rp_string = rp_data.read()

        chunks, eng_block = rp_string.split('[GEOCONV', 1)
        # chunks = chunks.split('(XYZ)')[1].split("\n\n")

        _, max_force = eng_block.split("energy", 1)[1].split("max-force", 1)
        # eng_block = eng_block.strip()
        max_force = max_force.split("max-step", 1)[0].strip()
        forces = np.array(max_force.splitlines()).astype('float')
        return forces

def parse_reaction_dimer_forces(dimer_file):
    with open(dimer_file) as dimer_dump:
        dimer_dat = dimer_dump.read()
        rx_opt, dimer_dat = dimer_dat.split("MINMODE:METHOD", 1)
        try:
            dimer_dat, prod_dat = dimer_dat.rsplit("MinModeTranslate", 1)
        except:
            print(dimer_file)
            raise
        eng_dat = dimer_dat.strip().rsplit("\n", 1)[-1]
        bits = eng_dat.split()
        curve = bits[-2]
        force = bits[-4]

        rx_force = rx_opt.rsplit("BFGS", 1)[-1].strip().split("\n", 1)[0].split()[-1]
        prod_force = prod_dat.rsplit("BFGS", 1)[-1].strip().split("\n", 1)[0].split()[-1]

        return np.array([rx_force, force, curve, prod_force])


def parse_dimer_trajectory(dimer_file,
                           reactant_file,
                           ts_file,
                           product_file,
                           load_absolute_energies=True,
                           load_mols=True,
                           parse_struct=True):
    with open(dimer_file) as dimer_dump:
        dimer_dat = dimer_dump.read()
        eng_dat = dimer_dat.split("Absolute Energy", 1)
        if len(eng_dat) == 1:
            print("Bad ouput:", dimer_file)
            return None
        engs = np.array(parsers.Number.findall(eng_dat[1])).astype(float)
        abs_eng, engs = engs[:3], engs[3:]
        if len(engs) == 2:
            # parser misses `0.` I guess
            engs = np.concatenate([[0.], engs])
    if parse_struct:
        if load_mols:
            geoms = [
                Molecule.from_file(reactant_file, energy_evaluator='aimnet2'),
                Molecule.from_file(ts_file, energy_evaluator='aimnet2'),
                Molecule.from_file(product_file, energy_evaluator='aimnet2'),
            ]
        else:
            geoms = []
            for f in [reactant_file, ts_file, product_file]:
                with open(f) as dump:
                    xyz = dump.read().split("\n", 2)[-1]
                    geoms.append({
                        'atoms': parsers.Word.findall(xyz),
                        'coords': np.array(parsers.Number.findall(xyz)).astype('float').reshape(-1, 3)
                    })
    else:
        geoms = []
        for f in [reactant_file, ts_file, product_file]:
            with open(f) as dump:
                geoms.append(dump.read())
    return ReactionProfileData(engs, geoms, abs_eng)

def parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=False,
                        parse_struct=True,
                        load_mols=True):
    conv = conf_path(reaction_class, idx_spec, 'opt_converged_000.xyz')
    if not os.path.isfile(conv):
        conv = conf_path(reaction_class, idx_spec, 'dimer.out')
        if not os.path.isfile(conv): return None
        return parse_dimer_trajectory(conv,
                                      conf_path(reaction_class, idx_spec, 'Reactant_force.xyz'),
                                      conf_path(reaction_class, idx_spec, 'TS_force.xyz'),
                                      conf_path(reaction_class, idx_spec, 'Product_force.xyz'),
                                      load_absolute_energies=load_absolute_energies,
                                      load_mols=load_mols,
                                      parse_struct=parse_struct)
    else:
        return parse_reaction_trajectory(conv,
                                         load_absolute_energies=load_absolute_energies,
                                         load_mols=load_mols,
                                         parse_struct=parse_struct)

def parse_reaction_forces(reaction_class, idx_spec):
    conv = conf_path(reaction_class, idx_spec, 'opt_converged_000.xyz')
    if not os.path.isfile(conv):
        conv = conf_path(reaction_class, idx_spec, 'dimer.out')
        if not os.path.isfile(conv): return None
        return parse_reaction_dimer_forces(conv)
    else:
        return parse_reaction_trajectory_forces(conv)

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
def load_reaction_index_data(reaction_class, idx_spec):
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

def animate_reaction(geoms, animation_file=None):
    anim = geoms[0].plot(
        geoms[0].embed_coords([g.coords for g in geoms]),
        animation_options=dict(animation_duration=5),
        include_save_buttons=True,
        image_size=800
    )

    if animation_file is not None:
        return anim.to_widget().write(animation_file)
    else:
        return anim

def animate_reaction_from_index(reaction_class, idx_spec, animation_file=True):
    geoms = parse_reaction_path(reaction_class, idx_spec)
    anim_name = "_".join(['anim', reaction_class] + list(idx_spec))
    if animation_file is True:
        animation_file = anim_name + '.html'
    return animate_reaction(geoms, animation_file=animation_file)

def plot_forces_from_index(reaction_class, idx_spec, structure_index=0, scaling=10, **opts):
    geom_test = parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=False)
    reaction_test = load_reaction_index_data(reaction_class, idx_spec)

    return geom_test.geometries[structure_index].plot(
        backend='x3d',
        mode_vectors=reaction_test.force * scaling,
        **opts
    )


def get_diene_embedding(geom, c1, c2):
    check_bonds = set(
        tuple(sorted(b[:2]))
        for b in geom.bonds
        if geom.atoms[b[0]] == "C" and geom.atoms[b[1]] == "C"
    )
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

def get_diene_atoms(reactant, product):
    graph_edits = reactant.edge_graph.graph_difference(product.edge_graph)
    inds = reactant.fragment_indices
    if len(inds) == 1:
        reactant, product = product, reactant
        inds = reactant.fragment_indices
        if len(inds) == 1: return None
    i = 0 if len(inds[0]) > len(inds[1]) else 1
    j = 1 if i == 0 else 0
    new_bond_pos = np.intersect1d(inds[i], np.unique(graph_edits[0]))
    if len(new_bond_pos) != 2:
        new_bond_pos = np.intersect1d(inds[j], np.unique(graph_edits[0]))

    if len(new_bond_pos) != 2:
        return None
    else:
        atoms = get_diene_embedding(reactant, *new_bond_pos)
        if atoms is None:
            new_bond_pos = np.intersect1d(inds[j], np.unique(graph_edits[0]))
            if len(new_bond_pos) > 2:
                return None
            atoms = get_diene_embedding(reactant, *new_bond_pos)
        return atoms


def compare_force_geometries(geoms_force, geoms_no_force,
                             embedding_atoms=None,
                             structure_index=0,
                             **opts
                             ):
    if isinstance(geoms_force, Molecule):
        geom = geoms_force
        geom_nf = geoms_no_force
    else:
        geom = geoms_force[structure_index]
        geom_nf = geoms_no_force[structure_index]
        if embedding_atoms is None:
            embedding_atoms = get_diene_atoms(geoms_force[0], geoms_force[-1])

    if embedding_atoms is not None:
        embedding_atoms = list(sorted(embedding_atoms))
    geom_f = geom.get_embedded_molecule(load_properties=False)
    geom_nf = geom_nf.get_embedded_molecule(ref=geom_f, load_properties=False, sel=embedding_atoms)
    g1 = geom_f.plot(backend='x3d', **opts)
    geom_nf.plot(figure=g1, backend='x3d', highlight_atoms=list(range(len(geom_nf.atoms))))
    return g1

def compare_force_geometries_from_index(reaction_class, idx_spec, embedding_atoms=None, structure_index=0, **opts):
    geom_test = parse_reaction_path(reaction_class, idx_spec, load_absolute_energies=False)
    geom_test_nf = parse_reaction_path(reaction_class, idx_spec[:-1], load_absolute_energies=False)

    return compare_force_geometries(geom_test.geometries,
                                    geom_test_nf.geometries,
                                    embedding_atoms=embedding_atoms,
                                    structure_index=structure_index,
                                    **opts)

def get_pair_rmsd(a, b):
    diff = np.asanyarray(a) - np.asanyarray(b)
    return np.linalg.norm(diff.flatten(), axis=-1) / np.sqrt(len(diff.flatten()))

def calculate_rmsd_reaction_coordiante(coords, rescale=True):
    diffs = np.diff(coords, axis=0)
    rmsds = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=-1) / np.sqrt(3 * len(coords[0]))
    cs = np.cumsum([0] + list(rmsds))
    if rescale:
        cs = nput.vec_rescale(cs)
    return cs

def get_incremental_RMSD(geometries, rescale=True):
    coords = geometries[0].embed_coords(np.array([g.coords for g in geometries]))
    return calculate_rmsd_reaction_coordiante(coords, rescale=rescale)

def get_reactant_ts_pos(engs):
    ts_pos = np.argmax(engs)
    if ts_pos == 0: return 0, 0
    if ts_pos < len(engs) - 1:
        p1 = np.argmin(engs[:ts_pos])
        p2 = ts_pos + 1 + np.argmin(engs[ts_pos + 1:])
        if engs[p1] < p2:
            return p2, ts_pos
        else:
            return p1, ts_pos
    else:
        return np.argmin(engs[:ts_pos]), ts_pos

def calculate_reactant_energy(engs):
    ts_pos = np.argmax(engs)
    if ts_pos == 0: return 0
    if ts_pos < len(engs) - 1:
        rx = max(np.min(engs[:ts_pos]), np.min(engs[ts_pos + 1:]))
    else:
        rx = np.min(engs[:ts_pos])
    return rx

def calculate_reaction_barrier(engs):
    ts_pos = np.argmax(engs)
    if ts_pos == 0: return 0
    if ts_pos < len(engs) - 1:
        rx = max(np.min(engs[:ts_pos]), np.min(engs[ts_pos + 1:]))
    else:
        rx = np.min(engs[:ts_pos])
    return engs[ts_pos] - rx
def get_true_reaction_barrier(reaction_class, idx_spec):
    engs = parse_reaction_path(reaction_class, idx_spec, parse_struct=False).energies
    return calculate_reaction_barrier(engs)

def check_reaction_energies(engs):
    if engs is None: return False
    engs = engs.energies
    ts_pos = np.argmax(engs)
    if ts_pos == 0: return False
    t1 = np.min(engs[:ts_pos])
    if ts_pos < len(engs) - 1:
        t2 = np.min(engs[ts_pos+1:])
        if t1 > t2:
            return t1 >= engs[0]
        else:
            return t2 >= engs[-1]
    else:
        return t1 >= engs[0]


def check_reaction_energies_from_index(reaction_class, idx_spec):
    engs = parse_reaction_path(reaction_class, idx_spec, parse_struct=False)
    return check_reaction_energies(engs)

def check_reaction_class(reaction_class, top_level=True):
    bad_rxns = []
    for x in itut.delete_duplicates(y[:2] for y in reaction_index_iter(reaction_class, return_conf=top_level)):
        if not check_reaction_energies_from_index(reaction_class, x):
            print(x)
            bad_rxns.append(x)
    file = f'{reaction_class}_bad_reactions.txt' if not top_level else f'{reaction_class}_bad_toplevel.txt'
    with open(file, 'w+') as bad:
        print(pprint.pformat(bad_rxns), file=bad)
    return bad_rxns

def plot_reaction_profiles_from_index(reaction_class, idx_spec,
                                      absolute_energies=None,
                                      **opts):
    load_absolute_energies = absolute_energies is None
    geom_test = parse_reaction_path(reaction_class, idx_spec,
                                    load_absolute_energies=load_absolute_energies)
    geom_test_nf = parse_reaction_path(reaction_class, idx_spec[:2],
                                       load_absolute_energies=load_absolute_energies)

    return plot_reaction_profiles(geom_test, geom_test_nf, absolute_energies=absolute_energies, **opts)

def remove_transrot(mol:Molecule, mode):
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

def compile_reaction_class(reaction_class, include_toplevel=False, return_conf=False):
    inds = []
    atom = []
    geoms = []
    engs = []
    for ind in reaction_index_iter(reaction_class, include_toplevel=include_toplevel, return_conf=return_conf):
        rpd:ReactionProfileData = parse_reaction_path(reaction_class, ind, load_mols=False)
        if rpd is not None:
            inds.append([int(x) for x in ind])
            atom.append(rpd.geometries[0]['atoms'])
            geoms.append(
                np.concatenate([g['coords'] for g in rpd.geometries], axis=0)
                * UnitsData.convert("Angstroms", "BohrRadius")
            )
            engs.append(rpd.energies)

    return {
        'inds':inds,
        'atoms':atom,
        'coords':geoms,
        'energies':engs
    }

def compile_reaction_forces(reaction_class, include_toplevel=False, return_conf=False):
    inds = []
    forces = []
    for ind in reaction_index_iter(reaction_class, include_toplevel=include_toplevel, return_conf=return_conf):
        force = parse_reaction_forces(reaction_class, ind)
        if force is not None:
            inds.append([int(x) for x in ind])
            forces.append(force)

    return {
        'inds':inds,
        'forces':forces
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


def compute_original_barriers(agg):
    engs = agg['energies']
    barriers = np.zeros(len(engs))
    for i,eng in enumerate(engs):
        ts_pos = np.argmax(eng)
        if ts_pos == 0:
            barriers[i] = 0
        else:
            barriers[i] = eng[ts_pos] - eng[0]
    return barriers

def compute_aggregate_barriers(agg):
    engs = agg['energies']
    barriers = np.zeros(len(engs))
    for i,eng in enumerate(engs):
        ts_pos = np.argmax(eng)
        if ts_pos == 0:
            barriers[i] = 0
        else:
            if ts_pos < len(eng) - 1:
                rx = max(np.min(eng[:ts_pos]), np.min(eng[ts_pos + 1:]))
            else:
                rx = np.min(eng[:ts_pos])
            barriers[i] = eng[ts_pos] - rx
    return barriers

mass_mapping = {}
def _compute_molar_mass(atoms, mass_mapping=mass_mapping):
    return sum(mass_mapping[a] for a in atoms)
def _update_molar_masses(agg, mass_mapping=mass_mapping):
    for a in np.unique(np.concatenate(agg['atoms'], axis=0)):
        if a not in mass_mapping:
            mass_mapping[a] = AtomData[a, "Mass"]
def compute_molar_masses(agg):
    _update_molar_masses(agg)
    return np.array([_compute_molar_mass(a) for a in agg['atoms']])


def get_aggregate_index_map(agg):
    if 'index_map' not in agg:
        agg['index_map'] = {
            tuple(k):i
            for i,k in enumerate(agg['inds'])
        }
    return agg['index_map']

def find_index_position(agg, inds):
    return get_aggregate_index_map(agg)[tuple(inds)]

AggregateReactionData = collections.namedtuple('AggregateReactionData', ['index', 'r', 'energies', 'reactant', 'product', 'coords'])
def get_reaction_by_position(agg, pos):
    idx = agg['inds'][pos]
    atoms = agg['atoms'][pos]
    engs = agg['energies'][pos]
    geoms = np.reshape(agg['coords'][pos], (-1, len(atoms), 3))
    ts_pos = np.argmax(engs)
    if ts_pos == 0:
        rx_pos = 0
    else:
        rx_pos = np.argmin(engs[:ts_pos])
    reactant = Molecule(atoms, geoms[rx_pos])
    product = Molecule(atoms, geoms[-1])
    new_coords = reactant.embed_coords(geoms)
    return AggregateReactionData(
        idx,
        calculate_rmsd_reaction_coordiante(new_coords, rescale=False),
        engs,
        reactant,
        product,
        new_coords
    )

def get_aggregate_reaction_by_index(agg, idx):
    return get_reaction_by_position(agg, find_index_position(agg, idx))

def get_reaction_diene(rx_data:AggregateReactionData):
    return get_diene_atoms(rx_data.reactant, rx_data.product)

def get_reaction_cc_bond_lengths(rx_data:'AggregateReactionData|ReactionProfileData'):
    if hasattr(rx_data, 'reactant'):#:isinstance(rx_data, AggregateReactionData):
        ats = get_diene_atoms(rx_data.reactant, rx_data.product)
        coords = rx_data.coords
    else:
        ats = get_diene_atoms(rx_data.geometries[0], rx_data.geometries[-1])
        coords = np.array([g.coords for g in rx_data.geometries])
    return np.linalg.norm(np.diff(coords[:, ats, :], axis=1), axis=-1)

def plot_cc_bond_lengths(r, reaction_cc_lengths, figure=None, label_names=None, **opts):
    if label_names is None:
        label_names = ["C=C", "C\N{MINUS SIGN}C", "C=C"]
    for i, cc_lens in enumerate(reaction_cc_lengths):
        figure = plt.Plot(
            r * UnitsData.convert("BohrRadius", "Angstroms"),
            cc_lens * UnitsData.convert("BohrRadius", "Angstroms"),
            figure=figure,
            **collections.ChainMap(
                opts,
                dict(
                    plot_label="CC Bond Variation",
                    label=label_names[i],
                    aspect_ratio=1 / 1.61,
                    image_size=800,
                    axes_labels=["$R$ ($\\AA$)", "$r_\\text{CC}$ ($\\AA$)"],
                    plot_legend=True,
                    legend_style={'frameon': False, 'fontsize': 16}
                )
            )
        )
    return figure


def get_profile_coordinate(prof_data:'ReactionProfileData|AggregateReactionData', rescale=False):
    if hasattr(prof_data, 'r'):
        r = prof_data.r
        if rescale:
            r = nput.vec_rescale(r)
        return r
    else:
        return get_incremental_RMSD(prof_data.geometries, rescale=rescale)

def get_profile_energies(prof_data:'ReactionProfileData|AggregateReactionData'):
    if hasattr(prof_data, 'absolute_energies'):
        eng = prof_data.absolute_energies
        geometries = prof_data.geometries
    else:
        eng = None
        geometries = [
            prof_data.reactant.modify(coords=c, energy_evaluator='aimnet2')
            for c in prof_data.coords
        ]

    if eng is None:
        eng = get_absolute_energies(geometries)
    return eng

def plot_reaction_profiles(
        geom_test:'ReactionProfileData|AggregateReactionData',
        geom_test_nf:'ReactionProfileData|AggregateReactionData',
        absolute_energies=None,
        rescale=False,
        **opts):
    if absolute_energies is None:
        absolute_energies = [None, None]
    e1, e2 = absolute_energies
    if e1 is None:
        e1 = get_profile_energies(geom_test)
    if e2 is None:
        e2 = get_profile_energies(geom_test_nf)
    conv = UnitsData.convert("BohrRadius", "Angstroms") if not rescale else 1
    unit = "$\\AA$" if not rescale else "arb."
    if len(e1) == 3:
        fig1 = plt.Plot(
            get_profile_coordinate(geom_test_nf, rescale=rescale) ,
            e2 - e2[0],
            # aspect_ratio=3/4,
            # axes_labels=["R", 'Energy (kcal mol$^{-1}$)'],
            **collections.ChainMap(
                opts,
                dict(
                    aspect_ratio=1 / 1.61,
                    image_size=800,
                    axes_labels=[f"$R$ ({unit})", 'Energy (kcal mol$^{-1}$)'],
                    label='No Force',
                    plot_legend=True,
                    legend_style={
                        'frameon': False,
                        'loc': 'upper left'
                    }
                )
            ),
        )
        # plt.HorizontalLinePlot(
        #     x,
        #     f,
        #     figure=figure,
        #     color=c,
        #     plot_range=plot_range,
        #     ticks=ticks,
        #     **style_dict
        # )
        r = get_profile_coordinate(geom_test, rescale=rescale) * conv
        pad = (r[-1] - r[0]) * .05
        for i,(x,y) in enumerate(zip(r, e1 - e2[0])):
            if i == 0:
                x = [x, x+2*pad]
            elif i == 2:
                x = [x-2*pad, x]
            else:
                x = [x-pad, x+pad]
            plt.HorizontalLinePlot(
                x,
                [y],
                figure=fig1,
                color='#'+plt.ColorPalette("default").color_strings[1],
                label="Force" if i == 0 else None
            )
    else:
        fig1 = plt.Plot(
            get_profile_coordinate(geom_test, rescale=rescale) * conv,
            e1 - e2[0],
            **collections.ChainMap(
                opts,
                dict(
                    aspect_ratio=1/1.61,
                    image_size=800,
                    axes_labels=[f"$R$ ({unit})", 'Energy (kcal mol$^{-1}$)'],
                    label='Force',
                    plot_legend=True,
                    legend_style={
                        'frameon':False,
                        'loc':'upper left'
                    }
                )
            )
        )
        plt.Plot(
            get_profile_coordinate(geom_test_nf, rescale=rescale) * conv,
            e2 - e2[0],
            # aspect_ratio=3/4,
            # axes_labels=["R", 'Energy (kcal mol$^{-1}$)'],
            figure=fig1,
            label='No Force'
        )
    # fig1.savefig("profile_3_2_1.png")
    return fig1

def compute_barrier_changes(no_force_data, force_data):
    barrs_og = compute_aggregate_barriers(no_force_data)
    barrs_new = compute_aggregate_barriers(force_data)
    baseline_barriers = {
        tuple(i):b for i,b in zip(no_force_data['inds'], barrs_og)
    }
    changes = np.zeros(len(force_data['inds']))
    for i, (idx, barr) in enumerate(zip(force_data['inds'], barrs_new)):
        key = tuple(idx[:2])
        changes[i] = barr - baseline_barriers[key]
    return changes

def compute_comparative_descriptors(
        no_force_data, force_data,
        prep_no_force_data,
        prep_force_data,
        compute_value,
        compare_results
):
    if 'molecule_cache' not in no_force_data:
        no_force_data['molecule_cache'] = {}
    baseline_dict = {}
    com_changes = [None] * len(force_data['inds'])
    for i, (idx, coords, energies) in enumerate(
            zip(
                force_data['inds'],
                force_data['coords'],
                force_data['energies']
            )
    ):
        key = tuple(idx[:-1])
        if key not in baseline_dict:
            if key not in no_force_data['molecule_cache']:
                comp_pos = find_index_position(no_force_data, key)
                no_force_data['molecule_cache'][key] = get_reaction_by_position(no_force_data, comp_pos)
            rx_base = no_force_data['molecule_cache'][key]
            base_data = prep_no_force_data(rx_base)
            if base_data is None:
                print(f"Problem at {key}")
                continue
            baseline_dict[key] = (base_data, compute_value(*base_data))

        og_data, results = baseline_dict[key]
        case_data = prep_force_data(*og_data, coords, energies)
        com_changes[i] = compare_results(results, compute_value(*case_data))
    return com_changes

def _compute_com_distance(coords, energies, frag_inds, mass_scaling):
    coms_1 = np.tensordot(coords[:, frag_inds[0], :], mass_scaling[frag_inds[0],], axes=[1, 0])
    coms_2 = np.tensordot(coords[:, frag_inds[1], :], mass_scaling[frag_inds[1],], axes=[1, 0])
    dists = np.linalg.norm(coms_1 - coms_2, axis=-1)
    ts_pos = np.argmax(energies)
    if ts_pos > 0:
        gs_pos = np.argmin(energies[:ts_pos])
    else:
        gs_pos = 0
    return dists[ts_pos], dists[gs_pos]

def _prep_topline_data(rx_base):
    frag_inds = rx_base.reactant.fragment_indices
    if len(frag_inds) == 1:
        frag_inds = rx_base.product.fragment_indices
    m = np.array(rx_base.reactant.masses)
    scale = m / np.sum(m)

    return rx_base.coords, rx_base.energies, frag_inds, scale

def _prep_com_coords(og_coords, _, frag_inds, scale, new_coords, energies):
    coords = np.reshape(new_coords, (-1, og_coords.shape[-2], 3))
    return coords, energies, frag_inds, scale

def _compare_coms(og_coms, new_coms):
    return og_coms[0] - new_coms[0], og_coms[1] - new_coms[1]

def compute_com_changes(no_force_data, force_data):
    return compute_comparative_descriptors(
        no_force_data, force_data,
        _prep_topline_data,
        _prep_com_coords,
        _compute_com_distance,
        _compare_coms
    )


def _compute_cc_distance(coords, energies, ats):
    r, ts = get_reactant_ts_pos(energies)
    dists = np.linalg.norm(np.diff(coords[:, ats, :], axis=1), axis=-1)
    return dists[ts], dists[r]

def _prep_topline_cc_data(rx_base):
    ats = get_diene_atoms(rx_base.reactant, rx_base.product)
    if ats is None:
        return None
    return rx_base.coords, rx_base.energies, ats

def _prep_cc_coords(og_coords, _, ats, new_coords, energies):
    coords = np.reshape(new_coords, (-1, og_coords.shape[-2], 3))
    return coords, energies, ats

def _compare_ccs(og_coms, new_coms):
    return new_coms[0] - og_coms[0], new_coms[1] - og_coms[1]

def compute_cc_changes(no_force_data, force_data):
    return compute_comparative_descriptors(
        no_force_data, force_data,
        _prep_topline_cc_data,
        _prep_cc_coords,
        _compute_cc_distance,
        _compare_ccs
    )

def plot_mm_activated_percentages(mm_groups, cutoff=0, **opts):
    masses, barriers = mm_groups
    return plt.BarPlot(
        np.arange(len(masses)),
        [100 * np.sum(u < cutoff) / len(u) for u in barriers],
        **collections.ChainMap(
            opts,
            dict(
                ticks=[
                    (
                        np.arange(len(masses)),
                        dict(labels=np.round(masses).astype(int))
                    ),
                    [0, 25, 50, 75, 100]
                ],
                plot_range=[None, [0, 100]],
                image_size=1000,
                aspect_ratio=1 / 1.61,
                axes_labels=["Molar mass (amu)", "Activation Percentage (%)"]
            )
        )
    )#.savefig('activated_percentage_plot.png')

def plot_mm_breakdown(mm_groups,
                      ncols=4,
                      spacings=(0, 0),
                      subimage_size=(300, 250),
                      **opts):
    masses, barriers = mm_groups
    bin_sort = np.flip(np.argsort([len(x) for x in barriers]))
    nrows = len(masses) // ncols + (0 if len(masses) % ncols == 0 else 1)
    fig = plt.GraphicsGrid(ncols=ncols, nrows=nrows, subimage_size=subimage_size, spacings=spacings)
    for i in range(len(bin_sort)):
        dats = barriers[bin_sort[i]]
        mm = masses[bin_sort[i]]
        row, col = np.unravel_index(i, (nrows, ncols))
        plt.HistogramPlot(dats,
                          **collections.ChainMap(
                              opts,
                              dict(
                                  bins=min(len(dats) // 4, 50),
                                  plot_range=[[-10, 10], [0, min(len(dats) // 2, 185)]],
                                  figure=fig[row, col],
                                  plot_label=f'Molar Mass {mm:.0f}'
                              )
                          ))
    return fig
    # fig.savefig('molar_mass_energies_nicer.png')


# def _prep_base_energies(rx_base):
#     ts_pos = np.argmax(rx_base.energies)
#     evaluator = rx_base.reactant.modify(
#         coords=rx_base.coords[ts_pos],
#         energy_evaluator='aimnet2'
#     )
#
#     return evaluator, rx_base.coords[ts_pos]
#
# def _prep_force_abs_energies(evaluator, og_coords, coords, energies):
#     ts_pos = np.argmax(energies)
#     coords = np.reshape(coords, (-1, og_coords.shape[-2], 3))
#     return evaluator, coords[ts_pos]
#
# def _compute_abs_engs(evaluator, coords):
#     return evaluator.calculate_energy(coords) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
#
# def _compute_ts_engs_diff(e1, e2):
#     return e1 - e2
#
# def _check_transition_state_eng_comps(no_force_data, force_data):
#     return compute_comparative_descriptors(
#         no_force_data, force_data,
#         _prep_base_energies,
#         _prep_force_abs_energies,
#         _compute_abs_engs,
#         _compute_ts_engs_diff
#     )

# def _prep_react_base_energies(rx_base):
#     engs = rx_base.energies
#     ts_pos = np.argmax(engs)
#     if ts_pos == 0:
#         r_pos = 0
#     elif ts_pos < len(engs) - 1:
#         p1 = np.argmin(engs[:ts_pos])
#         p2 = np.argmin(np.min(engs[ts_pos + 1:]))
#         if engs[p1] < engs[ts_pos+1+p2]:
#             r_pos = ts_pos+1+p2
#         else:
#             r_pos = p1
#     else:
#         r_pos = np.argmin(engs[:ts_pos])
#     evaluator = rx_base.reactant.modify(
#         coords=rx_base.coords[r_pos],
#         energy_evaluator='aimnet2'
#     )
#
#     return evaluator, rx_base.coords[ts_pos]
#
#
# def _prep_force_react_abs_energies(evaluator, og_coords, coords, engs):
#     ts_pos = np.argmax(engs)
#     if ts_pos == 0:
#         r_pos = 0
#     elif ts_pos < len(engs) - 1:
#         p1 = np.argmin(engs[:ts_pos])
#         p2 = np.argmin(np.min(engs[ts_pos + 1:]))
#         if engs[p1] < engs[ts_pos+1+p2]:
#             r_pos = ts_pos+1+p2
#         else:
#             r_pos = p1
#     else:
#         r_pos = np.argmin(engs[:ts_pos])
#     coords = np.reshape(coords, (-1, og_coords.shape[-2], 3))
#     return evaluator, coords[r_pos]
#
# def _check_reaction_eng_comps(no_force_data, force_data):
#     return compute_comparative_descriptors(
#         no_force_data, force_data,
#         _prep_react_base_energies,
#         _prep_force_react_abs_energies,
#         _compute_abs_engs,
#         _compute_ts_engs_diff
#     )

def _prep_base_energies(rx_base):
    ts_pos = np.argmax(rx_base.energies)
    evaluator = rx_base.reactant.modify(
        coords=rx_base.coords[ts_pos],
        energy_evaluator='aimnet2'
    ).get_energy_function()

    return evaluator, rx_base.coords#[ts_pos]

def _prep_force_abs_energies(evaluator, og_coords, coords, energies):
    # ts_pos = np.argmax(energies)
    coords = np.reshape(coords, (-1, og_coords.shape[-2], 3))
    return evaluator, coords

def _compute_abs_engs(evaluator, coords):
    return evaluator(coords) * UnitsData.convert("Hartrees", "Kilocalories/Mole")

def _return_energy_lists(e1, e2):
    return [e2, e1]
    # return e1 - e2

def get_absolute_energy_comparisons(no_force_data, force_data):
    energy_lists = compute_comparative_descriptors(
        no_force_data, force_data,
        _prep_base_energies,
        _prep_force_abs_energies,
        _compute_abs_engs,
        _return_energy_lists
    )
    e1 = [e[0] for e in energy_lists]
    e2 = [e[1] for e in energy_lists]
    return {'no_force_energies':e1, 'force_energies':e2}

def check_reaction_forces(energy_map, force_data,
                          force_max=2e-5,
                          force_max_dimer=5e-4,
                          curve_max=-2):
    return [
        abs(force[np.argmax(engs)]) < force_max
            if len(force) != 4 else
        (abs(force[1]) < force_max_dimer and force[2] < curve_max)
        for engs, force in zip(energy_map["energies"], force_data["forces"])
    ]

def get_energy_range_mask(engs, min, max, use_abs=False):
    engs = np.asanyarray(engs)
    if use_abs: engs = np.abs(engs)
    return np.logical_and(engs < max, engs > min)

def valid_mask_percent(engs, min, max, use_abs=False):
    return np.sum(get_energy_range_mask(engs, min, max, use_abs=use_abs)) / len(engs)

def get_ts_diffs(abs_engs):
    return np.array(
        [
            np.max(f) - np.max(nf)
            for f, nf in zip(abs_engs['force_energies'], abs_engs['no_force_energies'])
        ]
    )

def get_gs_diffs(abs_engs):
    return np.array([
        calculate_reactant_energy(f) - calculate_reactant_energy(nf)
        for f, nf in zip(abs_engs['force_energies'], abs_engs['no_force_energies'])
    ])

def get_valid_reaction_mask(
        abs_engs,
        force_data,
        forces,
        max_ediff=.5,
        min_ediff=-5
):
    ts_diffs = get_ts_diffs(abs_engs)
    gs_diffs = get_gs_diffs(abs_engs)
    force_checks = check_reaction_forces(force_data, forces)

    return np.logical_and(
        np.logical_and(
            force_checks,
            get_energy_range_mask(
                ts_diffs,
                min=min_ediff,
                max=max_ediff
            )
        ),
        get_energy_range_mask(
            gs_diffs,
            min=min_ediff,
            max=max_ediff
        )
    )


def get_good_barrier_points(no_force_data, force_data, forces, abs_eng, max_ediff=.1, min_ediff=-5,
                            return_mask=False
                            ):
    check_mask_1 = get_valid_reaction_mask(
        abs_eng,
        force_data,
        forces,
        max_ediff=max_ediff,
        min_ediff=min_ediff
    )
    check_inds1 = np.where(check_mask_1)[0]
    barr_diffs1 = compute_barrier_changes(no_force_data, force_data)
    barrs_1 = compute_aggregate_barriers(force_data)
    barr_diffs1 = barr_diffs1[check_inds1]
    barrs_1 = barrs_1[check_inds1]
    xy = (barrs_1 - barr_diffs1, barr_diffs1)
    if return_mask:
        return check_inds1, xy
    else:
        return xy

def plot_total_scatter(total_og, total_diff,
                       cutoff=.1,
                       figure=None,
                       upper_color=None,
                       plot_lines=True,
                       plot_range=None,
                       **opts):
    if plot_range is None:
        plot_range = [[np.min(total_og) - 2, np.max(total_og) + 2], [-5.5, 5.5]]
    mask1 = total_diff <= cutoff
    if mask1.any():

        scatter_base = plt.ScatterPlot(
            total_og[total_diff <= cutoff],
            total_diff[total_diff <= cutoff],
            **collections.ChainMap(
                opts,
                dict(
                    axes_labels=["E$_a^{\\text{(solv.)}}$ (kcal mol$^{-1}$)", "$\\Delta$E$_a$ (kcal mol$^{-1}$)"],
                    image_size=800,
                    figure=figure,
                    plot_range=plot_range
                )
            )
        )
        plt.ScatterPlot(
            total_og[total_diff > cutoff],
            total_diff[total_diff > cutoff],
            # axes_labels=["E$_a^{\\text{(solv.)}}$ (kcal mol$^{-1}$)", "$\\Delta$E$_a$ (kcal mol$^{-1}$)"],
            # image_size=800,
            color="#500000" if upper_color is None else upper_color,
            # plt.ColorPalette.color_lighten(plt.ColorPalette("default")[0], 2),
            figure=scatter_base
        )
    else:
        scatter_base = plt.ScatterPlot(
            total_og,
            total_diff,
            **collections.ChainMap(
                dict(color="#500000" if upper_color is None else upper_color),
                opts,
                dict(
                    axes_labels=["E$_a^{\\text{(solv.)}}$ (kcal mol$^{-1}$)", "$\\Delta$E$_a$ (kcal mol$^{-1}$)"],
                    image_size=800,
                    figure=figure,
                    plot_range=[[np.min(total_og) - 2, np.max(total_og) + 2], [-5.5, 5.5]]
                )
            )
        )
    if plot_lines:
        plt.HorizontalLinePlot(
            plot_range[0],
            [0],
            figure=scatter_base,
            linestyle="dashed",
            color="gray",
            linewidth=2
        )
        plt.HorizontalLinePlot(
            plot_range[0],
            [-1.36, -2.72],
            figure=scatter_base,
            linestyle="dashed",
            color="red",
            linewidth=2
        )
        # scatter_base.savefig("scatter_filtered_up_to_6.png")

    return scatter_base