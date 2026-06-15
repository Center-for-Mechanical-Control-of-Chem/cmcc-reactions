import itertools
import multiprocessing
import functools
import traceback

import scipy.sparse
import collections
import glob

from McUtils.ExternalPrograms import RDMolecule
import McUtils.Devutils as dev
import McUtils.Numputils as nput
from Psience.Molecools import Molecule
from rdkit import Chem
import hashlib
import numpy as np
import os
from rdkit.rdBase import BlockLogs

from . import utils
from .trajectory_tools import refine_trajectory, create_trajectory_data, TrajectoryData, ReoptimizedTrajectoryData

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


# Find atom indices by map number
def _get_atom_idx(mol, map_num):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"Atom map number {map_num} not found in molecule.")
def _pop_hydrogen(ref_mol, new_mol, idx1, idx2):
    a1 = ref_mol.GetAtomWithIdx(idx1)
    implicit_hs = a1.GetNumImplicitHs()
    explicit_hs = a1.GetNumExplicitHs()
    if implicit_hs > 0:
        return False
    elif explicit_hs > 0:
        a1 = new_mol.GetAtomWithIdx(idx2)
        explicit_hs = a1.GetNumExplicitHs()
        a1.SetNumExplicitHs(explicit_hs - 1)
        return False
    else:
        # Find one explicit neighbor on the new mol
        a1 = new_mol.GetAtomWithIdx(idx2)
        for neighbor in a1.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                h_idx = neighbor.GetIdx()
                new_mol.RemoveAtom(h_idx)
                return True
        return False
def join_fragments(smiles1: str, smiles2: str, new_bonds,
                   cache=None,
                   resanitize=True,
                   add_implicit_hydrogens=False,
                   fallback_to_ordering=False,
                   decrement_hydrogens=True) -> str:
    if cache is None:
        cache = {}
    if smiles1 not in cache:
        mol = RDMolecule.parse_smiles(smiles1, remove_hydrogens=True, add_implicit_hydrogens=add_implicit_hydrogens)
        if mol is not None:
            map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
            map.pop(0, None)
        else:
            map = None
        cache[smiles1] = {'mol': mol, 'map': map}
    if smiles2 not in cache:
        mol = RDMolecule.parse_smiles(smiles2, remove_hydrogens=True, add_implicit_hydrogens=add_implicit_hydrogens)
        if mol is not None:
            map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
            map.pop(0, None)
        else:
            map = None
        cache[smiles2] = {'mol': mol, 'map': map}
    mol1 = cache[smiles1]['mol']
    mol2 = cache[smiles2]['mol']

    if mol1 is None:
        raise ValueError(f"bad SMILES {smiles1}")
    if mol1 is None:
        raise ValueError(f"bad SMILES {smiles2}")

    map1 = cache[smiles1]['map']
    map2 = cache[smiles2]['map']
    offset = mol1.GetNumAtoms()

    map2 = {m+offset: i+offset for m,i in map2.items()}

    # Combine both molecules into one (no bond yet)
    combined = Chem.CombineMols(mol1, mol2)
    editable = Chem.RWMol(combined)

    for b in new_bonds:
        if len(b) == 2:
            m1, m2 = b
            t = 1
        else:
            m1, m2, t = b
        if fallback_to_ordering:
            idx1 = map1.get(m1 + 1, m1)
            idx2 = map2.get(m2 + offset + 1, m2 + offset)
        else:
            idx1 = map1[m1 + 1]
            idx2 = map2[m2 + offset + 1]

        if nput.is_numeric(t):
            if t == 1:
                t = Chem.BondType.SINGLE
            elif t == 2:
                t = Chem.BondType.DOUBLE
            elif t == 3:
                t = Chem.BondType.TRIPLE
            elif 1 < t and t < 2:
                t = Chem.BondType.AROMATIC
            else:
                raise ValueError(f"Bond type {t} is not supported.")

        editable.AddBond(idx1, idx2, t)
        if decrement_hydrogens:
            success = _pop_hydrogen(mol1, editable, idx1, idx1)
            i2 = idx2 - offset
            if success:
                offset = offset
                idx2 = idx2 - 1
                map2 = {m:i-1 for m,i in map2.items()}
            _pop_hydrogen(mol2, editable, i2, idx2)
    joined = editable.GetMol()

    if resanitize:
        Chem.SanitizeMol(joined)

    for m,i in map1.items():
        joined.GetAtomWithIdx(i).SetAtomMapNum(m)
    for m,i in map2.items():
        joined.GetAtomWithIdx(i).SetAtomMapNum(m - offset + len(map1))

    if add_implicit_hydrogens is not None:
        joined = Chem.RemoveHs(joined)
    return Chem.MolToSmiles(joined)

def set_chiralities(base_smiles, site_chirality_map):
    if not isinstance(base_smiles, str):
        mol = Chem.Mol(base_smiles)
    else:
        mol = Chem.MolFromSmiles(base_smiles)
    atom_map_pos = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    atom_map_pos.pop(0, None)

    for map_num, winding in site_chirality_map.items():
        atom_idx = atom_map_pos[map_num]
        winding_map = {
            "CW": Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            "CCW": Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
        if winding.upper() not in winding_map:
            raise ValueError("winding must be 'CW' or 'CCW'")
        atom = mol.GetAtomWithIdx(atom_idx)


        atom.SetChiralTag(winding_map[winding.upper()])

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol)

def set_stereo(base_smiles, active_sites,  stereo):
    # inject stereo information in the RDKit graph
    if not isinstance(base_smiles, str):
        mol = Chem.Mol(base_smiles)
    else:
        mol = Chem.MolFromSmiles(base_smiles)
    atom_map_pos = {atom.GetAtomMapNum():atom.GetIdx() for atom in mol.GetAtoms()}
    atom_map_pos.pop(0, None)

    if nput.is_int(active_sites[0]):
        active_sites = [active_sites]
    for a,b,c,d in active_sites:
        i, j, k, l = atom_map_pos[a], atom_map_pos[b], atom_map_pos[c], atom_map_pos[d]
        # TODO: ensure this is robust, might need to iterate on thiz
        mol.GetBondBetweenAtoms(j, k).SetStereo(Chem.BondStereo.STEREOE if stereo == "trans" else Chem.BondStereo.STEREOZ)
        mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.ENDDOWNRIGHT)
        mol.GetBondBetweenAtoms(k, l).SetBondDir(Chem.BondDir.ENDUPRIGHT)
        set_stereo = False
        with BlockLogs():
            for a, b in [(j, k), (k, j)]:
                if set_stereo: break
                for c, d in [(i, l), (l, i)]:
                    # this is bad practice, I should look up what they are actually doing
                    # but we are going quick and dirty
                    try:
                        mol.GetBondBetweenAtoms(a, b).SetStereoAtoms(c, d)
                    except RuntimeError:
                        ...
                    else:
                        set_stereo = True
                        break
            else:
                raise ValueError(f"failed to set stereo atoms for {i},{j},{k},{l}")

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    smi = Chem.MolToSmiles(mol)
    return smi

def fragment_to_smiles_iterator(
        template,
        fragments,
        active_sites,
        chiralities=None,
        add_implicit_hydrogens='full'
):
    cache = {}
    nsites = len(active_sites)
    for frags in itertools.combinations_with_replacement(fragments, nsites):
        temp = template
        for site,frag in zip(active_sites, frags):
            try:
                temp = join_fragments(temp, frag, [[site, 0]],
                                      cache=cache,
                                      add_implicit_hydrogens=add_implicit_hydrogens)
            except Chem.rdchem.AtomValenceException:
                continue
        if chiralities is not None:
            chiralities = [
                [c] if isinstance(c, str) else c
                for c in chiralities
            ]
            for c_set in itertools.product(*chiralities):
                yield set_chiralities(temp, dict(zip(active_sites, c_set)))
        else:
            yield temp

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
        "evaluator",
        'optimization_settings'
    ],
    defaults=[None]
)
utils.register_namedtuple(InitialProductData)
def create_product_data(struct, inds, energy=None, smiles=None, energy_evaluator=None, optimization_settings=None):
    return InitialProductData(
        smiles=smiles,
        atoms=struct.atoms,
        coords=struct.coords,
        bonds=[[int(i), int(j), float(t)] for i, j, t in struct.bonds],
        energy=energy,
        breakpoints=inds,
        evaluator=energy_evaluator,
        optimization_settings=optimization_settings
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
    product_data = create_product_data(struct, inds, energy=energy, smiles=smiles, energy_evaluator=energy_evaluator)
    utils.write_namedtuple(
        os.path.join(output_dir, info_file),
        product_data
    )
    return product_data

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
                                    info_file='product.json',
                                    rmsd_cutoff=.025,
                                    preopt_iterations=50,
                                    verbose=False,
                                    callback=None
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
            smiles_cache.add(u_smiles)
            if output_dir is not None:
                if smiles_hash_generator is not None:
                    smiles_label = smiles_hash_generator(u_smiles)
                else:
                    smiles_label = str(smiles_index)
                if os.path.isfile(os.path.join(output_dir, smiles_label, 'conformer_info.json')):
                    print("Pre-Optimized SMILES: ", smiles, f"({smiles_label})")
                    if callback is not None:
                        for f in glob.glob(os.path.join(output_dir, smiles_label, '*', info_file)):
                            product_data = utils.read_namedtuple(f)
                            callback(product_data, f)
                    continue
            else:
                smiles_label = None
        else:
            smiles_label = None

        if verbose:
            if smiles_label is None:
                u_smiles = Chem.CanonSmiles(smiles)
                if smiles_hash_generator is not None:
                    smiles_label = smiles_hash_generator(u_smiles)
                else:
                    smiles_label = str(smiles_index)
            print("Processing SMILES: ", smiles, f"({smiles_label})")

        conf_gen_options = conf_gen_defaults | conf_gen_options
        actual_num_confs = conf_gen_options.pop('numConfs', conf_gen_defaults.pop('num_confs', num_structs))
        if not evaluate_energy:
            actual_num_confs = num_structs

        if hasattr(calc, 'process_output'):
            calc = {'method':'aimnet2', 'model':calc}

        structs:list[Molecule] = Molecule.from_string(smiles, 'smi',
                                                      num_confs=actual_num_confs,
                                                      energy_evaluator=calc,
                                                      conf_gen_options=conf_gen_options,
                                                      spin=1
                                                      )
        ref = structs[0].get_embedded_molecule()
        structs = [s.get_embedded_molecule(ref=ref) for s in structs]
        if rmsd_cutoff is not None:
            structs = get_rmsd_pruned_structs(structs, rmsd_cutoff=rmsd_cutoff)
        if preoptimize:
            if optimizer_settings is None:
                optimizer_settings = {}
            os2 = optimizer_settings | {'max_iterations':preopt_iterations}
            structs = [struct.optimize(**os2) for struct in structs]
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
                    energy_evaluator=energy_evaluator,
                    info_file=info_file
                )
                if callback is not None:
                    callback(product_data, os.path.join(output_dir, smiles_label, str(i), info_file))
            else:
                product_data = create_product_data(
                    struct, diene_inds,
                    energy=engs[i],
                    smiles=smiles,
                    energy_evaluator=energy_evaluator
                )
                if callback is not None:
                    callback(product_data, None)

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
def inchi_key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))#.replace("/", "_").replace("\\", "^")
def generate_products_and_optimize_from_iterator(
        base_iterator,
        conf_gen_options=None,
        take_unique=True,
        num_structs=10,
        calc=None,
        evaluate_energy=True,
        energy_evaluator='aimnet2',
        preoptimize=True,
        optimizer_settings=None,
        smiles_hash_generator='inchi',
        info_file='product.json',
        output_dir=None,
        parallelizer=None,
        batch_size=50,
        verbose=False,
        callback=None
):
    base_iterator = enumerate(base_iterator)
    if dev.str_is(smiles_hash_generator, 'inchi'):
        smiles_hash_generator = inchi_key
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
            info_file=info_file,
            verbose=verbose,
            callback=callback
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
                    info_file=info_file,
                    verbose=verbose,
                    callback=callback
                ),
                batches
            ):
                final_smiles.extend(f_smiles)
                final_structures.extend(f_struct)
                energies.extend(f_eng)
                indices.extend(f_ind)

        return final_smiles, final_structures, energies, indices

def generate_products_and_optimize_from_diene_templates(
        templates, diene_atoms, group_map,
        diene_template=diene_template,
        output_dir=None,
        **opt_args
):
    base_iterator = product_smiles_iterator(
        templates, diene_atoms, group_map,
        diene_template=diene_template
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
    return generate_products_and_optimize_from_iterator(
        base_iterator,
        **opt_args
    )

def generate_products_and_optimize(
        template,
        fragments,
        active_sites,
        chiralities=None,
        output_dir=None,
        max_products=None,
        **opt_args
):
    base_iterator = fragment_to_smiles_iterator(
        template, fragments, active_sites,
        chiralities=chiralities
    )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        dev.write_json(
            os.path.join(output_dir, 'templates.json'),
            {
                'template': template,
                'fragments': fragments,
                'active_sites': active_sites,
                'chiralities': chiralities
            }
        )
    if max_products is not None:
        base_iterator = itertools.islice(base_iterator, max_products)
    return generate_products_and_optimize_from_iterator(
        base_iterator,
        output_dir=output_dir,
        **opt_args
    )

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

default_driven_bonds = ((0, 2), (3, 1))
def generate_initial_reaction_sampling(mol,
                                       max_step=4,
                                       nsteps=30,
                                       max_iterations=500,
                                       extra_constraints=None,
                                       driven_bonds=None,
                                       **optimizer_settings):
    if driven_bonds is None:
        driven_bonds = default_driven_bonds

    driven_bonds = [tuple(b) for b in driven_bonds]
    # zm = mol.get_bond_zmatrix(
    #     initial_backbone=sum(driven_bonds, ()),
    #     required_coordinates=driven_bonds)
    zm = mol.break_bonds(driven_bonds).get_bond_zmatrix(
        required_coordinates=driven_bonds
    )

    int_mol = mol.modify(internals=zm)
    _, geoms, _ = int_mol.relaxed_scan(
        [0, max_step, nsteps],
        {
            b:1
            for b in driven_bonds
        },
        max_iterations=max_iterations,
        coordinate_constraints=extra_constraints,
        **optimizer_settings
        # coordinate_constraints=[(2, 4, 5, 3)]
        # region_constraints={
        #     (0,1,23)
        # }
    )

    return geoms



def generate_reactants_from_products(
        product_data: InitialProductData,
        max_iterations=150,
        nsteps=30,
        max_step=4,
        profile_generator='pys-ts',
        energy_evaluator='aimnet2',
        reoptimize_product=True,
        optimizer='pysis',
        optimizer_method='rfo',
        refine_endpoints=True,
        refine_ts=True,
        output_dir=None,
        info_file='trajectory.json',
        extra_constraints=None,
        **optimization_settings
) -> ReoptimizedTrajectoryData:
    mol = Molecule(
        product_data.atoms,
        product_data.coords,
        product_data.bonds,
        energy_evaluator=energy_evaluator,
        spin=1
    )
    if reoptimize_product:
        mol = mol.optimize(
            mode=optimizer,
            method=optimizer_method,
            max_iterations=max_iterations
        )

    init_traj = generate_initial_reaction_sampling(
        mol,
        max_step=max_step,
        nsteps=nsteps,
        max_iterations=max_iterations,
        driven_bonds=product_data.breakpoints,
        extra_constraints=extra_constraints,
        **optimization_settings
    )

    base_structs = [
        mol.modify(coords=t, energy_evaluator=energy_evaluator)
        for t in init_traj
    ]

    init_traj = create_trajectory_data(base_structs)

    if profile_generator is not None:
        try:
            new_traj = refine_trajectory(
                init_traj,
                init_traj,
                energy_evaluator=energy_evaluator,
                profile_generator=profile_generator,
                refine_endpoints=refine_endpoints,
                refine_ts=refine_ts,
                **optimization_settings
                )
        except Exception as e:
            print("FAILURE ON REFINEMENT:")
            traceback.print_exc()
            new_traj = init_traj
        # new_traj = reoptimize_trajectory(
        #     mol,
        #     init_traj,
        #     max_iterations=max_iterations,
        #     profile_generator=profile_generator,
        #     energy_evaluator=energy_evaluator,
        #     **optimization_settings
        # )
    else:
        new_traj = init_traj

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