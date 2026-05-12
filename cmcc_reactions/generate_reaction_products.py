import itertools
import multiprocessing
import functools
import scipy.sparse
import collections

#TODO: just use the mcutils functions for most of this...
import McUtils.Devutils as dev
import McUtils.Numputils as nput
from Psience.Molecools import Molecule
from Psience.Reactions import Reaction
from rdkit import Chem
# from rdkit.Chem import AllChem
import hashlib
import numpy as np
import os
# import ase
# import ase.optimize
# import ase.io

from . import utils

__all__ = [
    "generate_products_and_optimize",
    "generate_reactants_from_products"
]

diene_template = "[{D[0]}:1][{D[1]}]=[{D[2]}][{D[3]}:2]"
def create_diene_template(base_template, C1="C", C2="C", C3="C", C4="C"):
    return base_template.format(D=[C1, C2, C3, C4])

def modify_template(template:str, group_map:dict[str, str]):
    for k,v in group_map.items():
        template = template.replace('['+k+']', v)
    return template

def get_bond_breaking_indices(mol:str, bonds=((1,3), (2, 4))):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    am_inds = {}
    for a in mol.GetAtoms():
        mn = a.GetAtomMapNum()
        if mn > 0:
            if mn in am_inds:
                raise ValueError(f"duplicate atom map number, {mn}")
            am_inds[mn] = a.GetIdx()

    pairs = []
    for i,j in bonds:
        if i not in am_inds:
            raise ValueError(f"bond index {i} not in atom map")
        if j not in am_inds:
            raise ValueError(f"bond index {j} not in atom map")
        pairs.append(
            (am_inds[i], am_inds[j])
        )

    return pairs

def build_smiles(template, diene_atoms, group_map, diene_template=diene_template):
    diene_core = create_diene_template(diene_template, *diene_atoms)
    return modify_template(
        template,
        dict(group_map, diene=diene_core)
    )
def product_smiles_iterator(templates, diene_atom_lists, group_map, diene_template=diene_template):
    map_keys = group_map.keys()
    map_values = group_map.values()
    if isinstance(templates, str):
        templates = [templates]
    if isinstance(diene_atom_lists[0], str):
        diene_atom_lists = [diene_atom_lists]
    for template in templates:
        for diene_atoms in diene_atom_lists:
            diene_core = create_diene_template(diene_template, *diene_atoms)
            for v in itertools.product(*map_values):
                #TODO: precompile the template so I can feed it into .format?
                yield modify_template(
                    template,
                    dict(zip(map_keys, v), diene=diene_core)
                )


# def get_atoms(mol):
#     return [a.GetSymbol() for a in mol.GetAtoms()]
# conf_gen_defaults = dict(
#     numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True,
#     useBasicKnowledge=True, enforceChirality=True, numThreads=0
# )
# def gen_conformers(mol, **opts):
#     atoms = get_atoms(mol)
#     for conf_id in AllChem.EmbedMultipleConfs(mol, **dict(conf_gen_defaults, **opts)):
#         conf = mol.GetConformer(conf_id)
#         coord = conf.GetPositions().copy()
#         yield atoms, coord
#
# def get_ase_calculator():
#     from aimnet2calc import AIMNet2ASE
#     return AIMNet2ASE('aimnet2',charge=0)
#
# optimizer_defaults = dict(fmax=0.000027)
# def optimize_structure(ase_mol, **opts):
#     opt = ase.optimize.BFGS(ase_mol, logfile=None)
#     opt.run(**dict(optimizer_defaults, **opts))
#     return ase_mol
#
# def get_conformers_and_energies(mol, calc=None, preoptimize=False,
#                                 evaluate_energy=True,
#                                 optimizer_settings=None,
#                                 **conf_gen_opts):
#     if calc is None and evaluate_energy:
#         calc = get_ase_calculator()
#     structs = []
#     engs = []
#     for atoms, coords in gen_conformers(mol, **conf_gen_opts):
#         conformer_tmp = ase.Atoms(symbols=atoms, positions=coords)
#         if evaluate_energy:
#             conformer_tmp.calc = calc
#             if preoptimize:
#                 if optimizer_settings is None:
#                     optimizer_settings = {}
#                 conformer_tmp = optimize_structure(conformer_tmp, **optimizer_settings) #TODO: support scipy
#             energy = conformer_tmp.get_potential_energy()
#         else:
#             energy = [0]
#         structs.append(conformer_tmp)
#         engs.append(energy[0])
#     return structs, np.array(engs)

InitialProductData = collections.namedtuple(
    'InitialProductData',
    [
        'smiles',
        'atoms',
        'coords',
        'bonds',
        'energy',
        'breakpoints',
        "evaluator"
    ]
)
utils.register_namedtuple(InitialProductData)
def create_product_data(struct, inds, energy=None, smiles=None, energy_evaluator=None):
    return InitialProductData(
        smiles=smiles,
        atoms=struct.atoms,
        coords=struct.coords,
        bonds=[[int(i), int(j), float(t)] for i, j, t in struct.bonds],
        energy=energy,
        breakpoints=inds,
        evaluator=energy_evaluator
    )
def write_product_structure(output_dir, struct, inds,
                            # conf_file='conf.xyz',
                            energy=None,
                            energy_evaluator=None,
                            smiles=None,
                            # index_file='isomer.txt',
                            info_file='product.json',
                            # bond_line='BREAK {0[0]:.0f} {0[1]:.0f}'
                            ):
    os.makedirs(output_dir, exist_ok=True)
    # comment = f"Energy: {energy} | SMILES: {smiles}"
    # with open(os.path.join(output_dir, conf_file), 'w+') as xyz:
    #     ase.io.write(xyz, struct, format='xyz', comment=comment)
    # with open(os.path.join(output_dir, index_file), 'w+') as ind_out:
    #     ind_out.writelines([
    #         bond_line.format(inds[0]),
    #         bond_line.format(inds[1])
    #     ])
    product_data = create_product_data(struct, inds, energy=energy, smiles=smiles, energy_evaluator=energy_evaluator)
    utils.write_namedtuple(
        os.path.join(output_dir, info_file),
        product_data
    )
    return product_data
    # dev.write_json(
    #     os.path.join(output_dir, info_file),
    #     {
    #         'smiles': smiles,
    #         'atoms':list(struct.atoms),
    #         'coords': struct.coords.tolist(),
    #         'bonds': [[int(i), int(j), float(t)] for i,j,t in struct.bonds],
    #         'energy': energy,
    #         'breakpoints': inds,
    #     }
    # )

def _get_rmsd_groups(rmsd_blocks, group, rmsd_cutoff):
    r, c = np.triu_indices(len(group), k=1)
    pair_rmsds = rmsd_blocks[r, c]
    equiv = pair_rmsds < rmsd_cutoff
    graph = np.zeros((len(group), len(group)), dtype=bool)
    np.fill_diagonal(graph, True)
    graph[r[equiv], c[equiv]] = True
    graph[c[equiv], r[equiv]] = True
    ncomp, labels = scipy.sparse.csgraph.connected_components(graph, directed=False, return_labels=True)
    _, groups = nput.group_by(np.arange(len(labels)), labels)[0]

    # we now need to split groups by RMSD cutoff
    representatives = []
    for g in groups:
        if len(g) == 1:
            representatives.append(g[0])
        else:
            rmsd_block = rmsd_blocks[np.ix_(g, g)]
            rr, cc = np.array(list(itertools.combinations(g, 2))).T
            g_rmsds = rmsd_blocks[rr, cc]
            split_groups = []
            if np.max(g_rmsds) > 2 * rmsd_cutoff:
                # group needs to be split, so we lower the cutoff until this is satisfied
                representatives.extend(
                    _get_rmsd_groups(rmsd_block, g, rmsd_cutoff / 2)
                )
            else:
                split_groups.append(g)
            for g in split_groups:
                # pick the element with the smallest average deviation from the other
                # elements in the set
                rep = g[np.argmin(np.average(rmsd_block, axis=0))]
                representatives.append(rep)

    return [group[r] for r in representatives]

def get_rmsd_pruned_structs(structs, rmsd_cutoff=.1):
    if len(structs) == 1:
        return structs
    mw_coords = np.array([
        s.coords * np.sqrt(s.masses / np.sum(s.masses))[:, np.newaxis]
        for s in structs
    ])
    r, c = np.triu_indices(len(structs), k=1)
    diffs = mw_coords[r,] - mw_coords[c,]
    diffs = diffs.reshape((len(diffs), -1))
    pair_rmsds = np.linalg.norm(diffs, axis=-1) / np.sqrt(len(diffs))
    rmsds = np.zeros((len(structs), len(structs)), dtype=float)
    rmsds[r, c] = pair_rmsds
    rmsds[c, r] = pair_rmsds

    struct_inds = np.sort(_get_rmsd_groups(rmsds, np.arange(len(structs)), rmsd_cutoff))
    return [structs[i] for i in struct_inds]

def smiles_hash(canonical_smiles):
    return hashlib.md5(canonical_smiles.encode()).hexdigest()
conf_gen_defaults = dict(
    numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True,
    useBasicKnowledge=True, enforceChirality=True, numThreads=0
)
def _generate_products_and_optimize(smiles_iterator,
                                    *,
                                    conf_gen_options,
                                    take_unique,
                                    num_structs,
                                    calc,
                                    evaluate_energy,
                                    energy_evaluator,
                                    preoptimize,
                                    optimizer_settings,
                                    smiles_hash_generator,
                                    output_dir,
                                    rmsd_cutoff=.025,
                                    preopt_iterations=50,
                                    verbose=False
                                    ):
    final_structures = []
    products = []

    smiles_cache = set()
    if conf_gen_options is None:
        conf_gen_options = {}
    for smiles_index,smiles in smiles_iterator:
        if take_unique:
            u_smiles = Chem.CanonSmiles(smiles)
            if u_smiles in smiles_cache: continue
            if output_dir is not None:
                if smiles_hash_generator is not None:
                    smiles_label = smiles_hash_generator(u_smiles)
                else:
                    smiles_label = str(smiles_index)
                if os.path.isfile(os.path.join(output_dir, smiles_label, 'conformer_info.json')): continue
            smiles_cache.add(u_smiles)

        if verbose:
            print("Processing SMILES: ", smiles)

        conf_gen_options = conf_gen_defaults | conf_gen_options
        actual_num_confs = conf_gen_options.pop('numConfs', conf_gen_defaults.pop('num_confs', num_structs))
        if not evaluate_energy:
            actual_num_confs = num_structs

        if hasattr(calc, 'process_output'):
            calc = {'method':'aimnet2', 'model':calc}

        structs:list[Molecule] = Molecule.from_string(smiles, 'smi',
                                                      num_confs=actual_num_confs,
                                                      energy_evaluator=calc,
                                                      conf_gen_options=conf_gen_options
                                                      )
        ref = structs[0].get_embedded_molecule()
        structs = [s.get_embedded_molecule(ref=ref) for s in structs]
        if rmsd_cutoff is not None:
            structs = get_rmsd_pruned_structs(structs, rmsd_cutoff=rmsd_cutoff)
        if preoptimize:
            if optimizer_settings is None:
                optimizer_settings = {}
            os2 = optimizer_settings | {'max_iterations':preopt_iterations}
            structs = [struct.optimize(**optimizer_settings) for struct in structs]
            if rmsd_cutoff is not None:
                structs = get_rmsd_pruned_structs(structs, rmsd_cutoff=rmsd_cutoff)
        if evaluate_energy:
            engs = [m.calculate_energy() for m in structs]
        else:
            engs = [None] * num_structs

        diene_inds = ((0, 2), (1, 3))#get_bond_breaking_indices(smiles)
        if evaluate_energy:
            if len(engs) > num_structs:
                top_indices = np.argpartition(engs, num_structs)[:num_structs]
            else:
                top_indices = np.argsort(engs)
        else:
            top_indices = np.arange(min([len(structs), num_structs]))

        for i in top_indices:
            struct = structs[i]
            if not preoptimize and evaluate_energy:
                if optimizer_settings is None:
                    optimizer_settings = {}
                struct = struct.optimize(**optimizer_settings)
            if output_dir is not None:
                if smiles_hash_generator is not None:
                    smiles_label = smiles_hash_generator(Chem.CanonSmiles(smiles))
                else:
                    smiles_label = str(smiles_index)
                product_data = write_product_structure(
                    os.path.join(output_dir, smiles_label, str(i)),
                    struct, diene_inds,
                    energy=engs[i],
                    smiles=smiles,
                    energy_evaluator=energy_evaluator
                )
            else:
                product_data = create_product_data(
                    struct, diene_inds,
                    energy=engs[i],
                    smiles=smiles,
                    energy_evaluator=energy_evaluator
                )

            products.append(product_data)
            final_structures.append(struct)

        if output_dir is not None:
            if smiles_hash_generator is not None:
                smiles_label = smiles_hash_generator(Chem.CanonSmiles(smiles))
            else:
                smiles_label = str(smiles_index)
            if hasattr(engs, 'tolist'):
                engs = engs.tolist()
            if hasattr(top_indices, 'tolist'):
                top_indices = top_indices.tolist()
            dev.write_json(
                os.path.join(output_dir, smiles_label, 'conformer_info.json'),
                {
                    'smiles': smiles,
                    'conf_ids': top_indices,
                    'energies': engs
                }
            )

    return final_structures, products

def iter_batched(iterable, n):
    # https://stackoverflow.com/a/8290490
    while True:
        batch = tuple(itertools.islice(iterable, n))
        if not batch:
            return
        yield batch
def generate_products_and_optimize(
        templates, diene_atoms, group_map,
        diene_template=diene_template,
        conf_gen_options=None,
        take_unique=True,
        num_structs=10,
        calc=None,
        evaluate_energy=True,
        energy_evaluator='aimnet2',
        preoptimize=True,
        optimizer_settings=None,
        smiles_hash_generator=None,
        output_dir=None,
        parallelizer=None,
        batch_size=50,
        verbose=False,
):
    base_iterator = enumerate(
        product_smiles_iterator(
            templates, diene_atoms, group_map,
            diene_template=diene_template
        )
    )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        dev.write_json(
            os.path.join(output_dir, 'templates.json'),
            {
                'templates': templates,
                'diene_bond': diene_atoms,
                'groups': group_map,
                'diene_template': diene_template
            }
        )
    if parallelizer is None:
        return _generate_products_and_optimize(
            base_iterator,
            conf_gen_options=conf_gen_options,
            take_unique=take_unique,
            num_structs=num_structs,
            calc=energy_evaluator,
            evaluate_energy=evaluate_energy,
            energy_evaluator=energy_evaluator,
            preoptimize=preoptimize,
            optimizer_settings=optimizer_settings,
            smiles_hash_generator=smiles_hash_generator,
            output_dir=output_dir,
            verbose=verbose
        )
    else:
        if parallelizer is True:
            parallelizer = multiprocessing.Pool()
        batches = iter_batched(base_iterator, batch_size)


        final_smiles = []
        final_structures = []
        energies = []
        indices = []
        with parallelizer:
            for f_smiles, f_struct, f_eng, f_ind in parallelizer.map(
                functools.partial(
                    _generate_products_and_optimize,
                    conf_gen_options=conf_gen_options,
                    take_unique=take_unique,
                    num_structs=num_structs,
                    calc=calc,
                    evaluate_energy=evaluate_energy,
                    energy_evaluator=energy_evaluator,
                    preoptimize=preoptimize,
                    optimizer_settings=optimizer_settings,
                    smiles_hash_generator=smiles_hash_generator,
                    output_dir=output_dir,
                    verbose=verbose
                ),
                batches
            ):
                final_smiles.extend(f_smiles)
                final_structures.extend(f_struct)
                energies.extend(f_eng)
                indices.extend(f_ind)

        return final_smiles, final_structures, energies, indices


def scan_opt(start_struct, step_max, step_n, scan_ind,
             *,
             coordinate_constraints,
             displacement_exponent=1,
             displacement_function=None,
             internal_opt=True,
             **opt_args):
    pre_opt = []
    opt_uuh = [start_struct]
    if nput.is_int(scan_ind):
        scan_ind = [scan_ind]
    if displacement_function is not None:
        disps = displacement_function(step_max, step_n)
    else:
        disps = np.linspace(0, 1, step_n + 1)**displacement_exponent * step_max
        disps = np.broadcast_to(disps[:, np.newaxis], (len(disps), len(scan_ind)))

    diffs = np.diff(disps, axis=0)
    for d in diffs:
        new_coords = opt_uuh[-1].get_displaced_coordinates(
            np.array([d]),
            which=scan_ind,
            use_internals='reembed',
            strip_embedding=True
        )
        if internal_opt:
            mod_opts = dict(coords=new_coords[0])
        else:
            mod_opts = dict(coords=new_coords[0], internals=None)
        pre_opt.append(opt_uuh[-1].modify(**mod_opts))
        new_opt_coords = pre_opt[-1].optimize(
            coordinate_constraints=coordinate_constraints,
            **opt_args
        ).coords
        opt_uuh.append(opt_uuh[-1].modify(coords=new_opt_coords))
    return disps, pre_opt, opt_uuh

def create_breakpoint_zmat(mol, bonds=((0, 2), (1, 3)), type='dibond'):
    ahh_bork = mol.break_bonds(bonds)
    frags = ahh_bork.fragment_indices
    # we'll make the CP moiety always go first
    if 0 in frags[0]:
        fragment_ordering = [0, 1]
    else:
        fragment_ordering = [1, 0]
    zm0 = ahh_bork.get_bond_zmatrix(
        fragment_ordering=fragment_ordering,
        fragments=frags,
        attachment_points=dict(bonds[:1])
    )
    ats = [z[0] for z in zm0]
    bork2 = ats.index(bonds[0][0]) < ats.index(bonds[0][1])
    bork_bonds = dict(bonds) | {j: i for i, j in bonds}
    if bork2:
        bork_atoms = [bonds[0][1], bonds[1][1]]
    else:
        bork_atoms = [bonds[0][0], bonds[1][0]]

    if dev.str_is(type, 'dibond'):
        zm = []
        main_ref_lines = {}
        main_ref = None
        for z in zm0:
            if z[0] in bork_atoms:
                if main_ref is None:
                    main_ref = z[2:]
                else:
                    a = bork_bonds[z[0]]
                    z = [z[0]] + main_ref_lines.get(
                        a,
                        [a] + main_ref
                    )
            elif z[1] in bork_bonds:
                main_ref_lines[z[1]] = z[1:]
            zm.append(z)

        constraints = bonds
        for z in zm:
            if z[1] in bork_atoms:
                constraints = constraints + (tuple(z[:3]), tuple(z))
                break
    else:
        zm = zm0
        constraints = bonds[:1]
        for z in zm:
            if z[0] in bork_atoms:
                constraints = constraints + (tuple(z[:3]), tuple(z))
                break

    return zm, constraints

default_driven_bonds = ((0, 2), (1, 3))
def generate_initial_reaction_sampling(mol,
                                       max_step=4,
                                       nsteps=15,
                                       max_iterations=500,
                                       driven_bonds=None,
                                       **optimizer_settings):
    if driven_bonds is None:
        driven_bonds = default_driven_bonds

    driven_bonds = [tuple(b) for b in driven_bonds]
    zm = mol.get_bond_zmatrix(required_coordinates=driven_bonds)

    int_mol = mol.modify(internals=zm)
    _, geoms, _ = int_mol.relaxed_scan(
        [0, max_step, nsteps],
        {
            b:1
            for b in driven_bonds
        },
        max_iterations=max_iterations,
        **optimizer_settings
        # coordinate_constraints=[(2, 4, 5, 3)]
        # region_constraints={
        #     (0,1,23)
        # }
    )

    return geoms


def get_critical_points(trajectory, energies=None, initial='product'):
    if energies is None:
        energies = [g.calculate_energy() for g in trajectory]

    ts_idx = np.argmax(energies)
    if initial == 'product':
        react_idx = ts_idx + np.argmin(energies[ts_idx:])
        prod_idx = np.argmin(energies[:ts_idx])
    else:
        react_idx = np.argmin(energies[:ts_idx])
        prod_idx = ts_idx + np.argmin(energies[ts_idx:])
    return energies, (ts_idx, react_idx, prod_idx)

TrajectoryData = collections.namedtuple(
    "TrajectoryData",
    [
        "atoms",
        "coordinates",
        "energies",
        "rmsds",
        "evaluator"
    ]
)
utils.register_namedtuple(TrajectoryData)

def create_trajectory_data(structures, energies=None, energy_evaluator=None, rmsds=None):
    if energy_evaluator is None:
        energy_evaluator = structures[0].energy_evaluator
        if energies is None:
            energies = [g.calculate_energy() for g in structures]
    else:
        if energies is None:
            energies = [g.modify(energy_evaluator=energy_evaluator).calculate_energy() for g in structures]

    coords = [s.coords for s in structures]
    if rmsds is None:
        rmsds = nput.incremental_eckart_rmsd(coords, masses=structures[0].masses, mass_weighted=False)

    return TrajectoryData(
        atoms=structures[0].atoms,
        coordinates=coords,
        energies=energies,
        rmsds=rmsds,
        evaluator=energy_evaluator
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
        "final_rmsds",
        "initial_trajectory",
        "initial_energies",
        "initial_rmsds",
        "raw_pre_sampling",
        "raw_pre_energies",
        "evaluator"
    ]
)
utils.register_namedtuple(ReoptimizedTrajectoryData)

def create_refined_trajectory_data(initial_trajectory:TrajectoryData,
                                   final_trajectory:TrajectoryData,
                                   initial_energies=None,
                                   initial_rmsds=None,
                                   final_energies=None,
                                   final_rmsds=None,
                                   pre_sampling=None,
                                   pre_sampling_energies=None,
                                   energy_evaluator=None,
                                   ):
    if not hasattr(initial_trajectory, 'atoms'):
        initial_trajectory = create_trajectory_data(initial_trajectory,
                                                    energies=initial_energies,
                                                    rmsds=initial_rmsds,
                                                    energy_evaluator=energy_evaluator)
    if not hasattr(final_trajectory, 'atoms'):
        final_trajectory = create_trajectory_data(final_trajectory,
                                                    energies=final_energies,
                                                    rmsds=final_rmsds,
                                                    energy_evaluator=energy_evaluator)
    if pre_sampling is not None and not hasattr(pre_sampling, 'atoms'):
        pre_sampling = create_trajectory_data(pre_sampling,
                                              energies=pre_sampling_energies)
    return ReoptimizedTrajectoryData(
        atoms=initial_trajectory.atoms,
        final_trajectory=final_trajectory.coordinates,
        final_energies=final_trajectory.energies,
        final_rmsds=final_trajectory.rmsds,
        initial_trajectory=initial_trajectory.coordinates,
        initial_energies=initial_trajectory.energies,
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
def reoptimize_trajectory(mol,
                          init_traj,
                          max_iterations=500,
                          profile_generator='neb',
                          energy_evaluator='aimnet2',
                          **optimization_settings
                          ):
    base_structs = [
        mol.modify(coords=t, energy_evaluator=energy_evaluator)
        for t in init_traj
    ]
    init_engs, (ts, r, p) = get_critical_points(base_structs)
    if p < r:
        traj_structs = list(reversed(base_structs[p:r+1]))
        traj_engs = list(reversed(init_engs[p:r+1]))
    else:
        traj_structs = base_structs[r:p+1]
        traj_engs = init_engs[r:p+1]

    eeee = Reaction([traj_structs[0]], [traj_structs[-1]],
                    profile_generator=profile_generator,
                    energy_evaluator=energy_evaluator
                    )
    prof = eeee.get_profile_generator(
        energy_evaluator=energy_evaluator
    )
    new_geoms = prof.generate(
        base_images=traj_structs,
        max_iterations=max_iterations,
        **optimization_settings
    )

    new_rmsds = prof.evaluate_profile_distances(new_geoms, normalize=False)
    new_engs = prof.evaluate_profile_energies(new_geoms)
    old_rmsds = prof.evaluate_profile_distances(traj_structs, normalize=False)

    return ReoptimizedTrajectoryData(
        atoms=base_structs[0].atoms,
        final_trajectory=np.array([t.coords for t in new_geoms]),
        final_energies=new_engs,
        final_rmsds=new_rmsds,
        initial_trajectory=np.array([t.coords for t in traj_structs]),
        initial_energies=traj_engs,
        initial_rmsds=old_rmsds,
        raw_pre_sampling=init_traj,
        raw_pre_energies=init_engs,
        evaluator=energy_evaluator
    )

def generate_reactants_from_products(
        product_data: InitialProductData,
        max_iterations=500,
        profile_generator='neb',
        energy_evaluator='aimnet2',
        reoptimize_product=True,
        output_dir=None,
        info_file='trajectory.json',
        **optimization_settings
) -> ReoptimizedTrajectoryData:
    mol = Molecule(
        product_data.atoms,
        product_data.coords,
        product_data.bonds,
        energy_evaluator=energy_evaluator
    )
    if reoptimize_product:
        mol = mol.optimize(max_iterations=max_iterations)

    init_traj = generate_initial_reaction_sampling(
        mol,
        max_step=4,
        nsteps=15,
        max_iterations=max_iterations,
        driven_bonds=product_data.breakpoints
    )

    new_traj = reoptimize_trajectory(
        mol,
        init_traj,
        max_iterations=max_iterations,
        profile_generator=profile_generator,
        energy_evaluator=energy_evaluator,
        **optimization_settings
    )

    if output_dir is not None:
        utils.write_namedtuple(
            os.path.join(output_dir, info_file),
            new_traj
        )

    return new_traj

def refine_trajectory(product_data: InitialProductData|ReoptimizedTrajectoryData|TrajectoryData,
                      trajectory_data: ReoptimizedTrajectoryData|TrajectoryData = None,
                      energy_evaluator=None,
                      profile_generator='pys-dimer',
                      output_dir=None,
                      info_file='refined.json',
                      method_options=None,
                      climb=True,
                      ts_opt_generator=None,#'pys-dimer',
                      ts_opt_settings=None,
                      ts_opt_optimizer=None,
                      thresh='gau_tight',
                      refine_endpoints=False,
                      refine_ts=True,
                      optimizer_settings=None,
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
    if 'thresh' not in optimizer_settings:
        optimizer_settings['thresh'] = thresh

    traj = [
        Molecule(product_data.atoms,
                 c,
                 energy_evaluator=energy_evaluator)
        for c in (
            trajectory_data.final_trajectory
                if hasattr(trajectory_data, 'final_trajectory') else
            trajectory_data.coordinates
        )
    ]

    if refine_endpoints:
        # uh = traj[0]
        traj[0] = traj[0].optimize()
        # print(traj[0].calculate_energy() - uh.calculate_energy())
        # uh2 = traj[-1]
        traj[-1] = traj[-1].optimize()
        # print(traj[-1].calculate_energy() - uh2.calculate_energy())


    if refine_ts:
        rxn = Reaction([traj[0]], [traj[-1]])
        if method_options is None:
            method_options = {}
        prof = rxn.get_profile_generator(profile_generator,
                                         energy_evaluator=energy_evaluator,
                                         climb=climb,
                                         **method_options)
        new_images = prof.generate(base_images=traj,
                                   optimizer_settings=optimizer_settings,
                                   **calc_options)

        if ts_opt_generator is not None:
            rxn = Reaction([new_images[0]], [new_images[-1]])
            prof = rxn.get_profile_generator(ts_opt_generator,
                                             energy_evaluator=energy_evaluator,
                                             climb=climb,
                                             **method_options)
            if ts_opt_settings is None:
                ts_opt_settings = optimizer_settings | dict(optimizer=ts_opt_optimizer)
            new_images = prof.generate(base_images=new_images,
                                       **ts_opt_settings)
    else:
        new_images = traj

    new_coords = np.array([t.coords for t in new_images])
    new_traj = ReoptimizedTrajectoryData(
        atoms=product_data.atoms,
        final_trajectory=np.array([t.coords for t in new_images]),
        final_energies=[i.calculate_energy() for i in new_images],
        final_rmsds=nput.incremental_eckart_rmsd(new_coords, masses=new_images[0].masses, mass_weighted=False),
        initial_trajectory=(
            trajectory_data.final_trajectory
                if hasattr(trajectory_data, 'final_trajectory') else
            trajectory_data.coordinates
        ),
        initial_energies=(
            trajectory_data.final_energies
                if hasattr(trajectory_data, 'final_trajectory') else
            trajectory_data.energies
        ),
        initial_rmsds=(
            trajectory_data.final_rmsds
                if hasattr(trajectory_data, 'final_trajectory') else
            trajectory_data.rmsds
        ),
        raw_pre_sampling=(
            trajectory_data.raw_pre_sampling
                if hasattr(trajectory_data, 'final_trajectory') else
            None
        ),
        raw_pre_energies=(
            trajectory_data.raw_pre_energies
                if hasattr(trajectory_data, 'final_trajectory') else
            None
        ),
        evaluator=energy_evaluator
    )

    if output_dir is not None:
        utils.write_namedtuple(
            os.path.join(output_dir, info_file),
            new_traj
        )

    return new_traj

def test_refinement_methods(
        product_data: InitialProductData, trajectory_data: ReoptimizedTrajectoryData,
        refiment_lists:dict[str, dict],
        info_file_template='refined-{name}.json',
        **global_options
):
    for name, opts in refiment_lists.items():
        info_file = info_file_template.format(name=name)
        refine_trajectory(product_data, trajectory_data,
                          **(
                              {'info_file':info_file}
                              | global_options
                              | opts
                          ))