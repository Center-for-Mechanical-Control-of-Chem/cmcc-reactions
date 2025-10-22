import itertools
import multiprocessing
import functools
from rdkit import Chem
from rdkit.Chem import AllChem
import hashlib
import numpy as np
import os
import ase
import ase.optimize
import ase.io

diene_template = "[{D[0]}:1]1[{D[1]}]=[{D[2]}][{D[3]}:2]"
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


def get_atoms(mol):
    return [a.GetSymbol() for a in mol.GetAtoms()]
conf_gen_defaults = dict(
    numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True,
    useBasicKnowledge=True, enforceChirality=True, numThreads=0
)
def gen_conformers(mol, **opts):
    atoms = get_atoms(mol)
    for conf_id in AllChem.EmbedMultipleConfs(mol, **dict(conf_gen_defaults, **opts)):
        conf = mol.GetConformer(conf_id)
        coord = conf.GetPositions().copy()
        yield atoms, coord

def get_ase_calculator():
    from aimnet2calc import AIMNet2ASE
    return AIMNet2ASE('aimnet2',charge=0)

optimizer_defaults = dict(fmax=0.000027)
def optimize_structure(ase_mol, **opts):
    opt = ase.optimize.BFGS(ase_mol, logfile=None)
    opt.run(**dict(optimizer_defaults, **opts))
    return ase_mol

def get_conformers_and_energies(mol, calc=None, preoptimize=False,
                                evaluate_energy=True,
                                optimizer_settings=None,
                                **conf_gen_opts):
    if calc is None and evaluate_energy:
        calc = get_ase_calculator()
    structs = []
    engs = []
    for atoms, coords in gen_conformers(mol, **conf_gen_opts):
        conformer_tmp = ase.Atoms(symbols=atoms, positions=coords)
        if evaluate_energy:
            conformer_tmp.calc = calc
            if preoptimize:
                if optimizer_settings is None:
                    optimizer_settings = {}
                conformer_tmp = optimize_structure(conformer_tmp, **optimizer_settings) #TODO: support scipy
            energy = conformer_tmp.get_potential_energy()
        else:
            energy = [0]
        structs.append(conformer_tmp)
        engs.append(energy[0])
    return structs, np.array(engs)

def write_structure(output_dir, struct, inds,
                    conf_file='conf.xyz', energy=None, smiles=None,
                    index_file='isomer.txt', bond_line='BREAK {0[0]:.0f} {0[1]:.0f}'):
    os.makedirs(output_dir, exist_ok=True)
    comment = f"Energy: {energy} | SMILES: {smiles}"
    with open(os.path.join(output_dir, conf_file), 'w+') as xyz:
        ase.io.write(xyz, struct, format='xyz', comment=comment)
    with open(os.path.join(output_dir, index_file), 'w+') as ind_out:
        ind_out.writelines([
            bond_line.format(inds[0]),
            bond_line.format(inds[1])
        ])

def smiles_hash(canonical_smiles):
    return hashlib.md5(canonical_smiles.encode()).hexdigest()
def _generate_products_and_optimize(smiles_iterator,
                                    *,
                                    conf_gen_options,
                                    take_unique,
                                    num_structs,
                                    calc,
                                    evaluate_energy,
                                    preoptimize,
                                    optimizer_settings,
                                    smiles_hash_generator,
                                    output_dir
                                    ):
    final_smiles = []
    final_structures = []
    energies = []
    indices = []

    smiles_cache = set()
    if conf_gen_options is None:
        conf_gen_options = {}
    if calc is None and evaluate_energy:
        calc = get_ase_calculator()
    for smiles_index,smiles in smiles_iterator:
        if take_unique:
            u_smiles = Chem.CanonSmiles(smiles)
            if u_smiles in smiles_cache: continue
            if output_dir is not None and smiles_hash_generator is not None:
                smiles_label = smiles_hash_generator(u_smiles)
                if os.path.isdir(os.path.join(output_dir, smiles_label)): continue
            smiles_cache.add(u_smiles)

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        structs, engs = get_conformers_and_energies(
            mol,
            calc=calc,
            preoptimize=preoptimize,
            optimizer_settings=optimizer_settings,
            evaluate_energy=evaluate_energy,
            **conf_gen_options
        )

        diene_inds = get_bond_breaking_indices(mol)
        top_indices = np.argpartition(engs, num_structs)[:num_structs]
        for i in top_indices:
            struct = structs[i]
            if not preoptimize and evaluate_energy:
                if optimizer_settings is None:
                    optimizer_settings = {}
                struct = optimize_structure(struct, **optimizer_settings)
            if output_dir is not None:
                if smiles_hash_generator is not None:
                    smiles_label = smiles_hash_generator(Chem.CanonSmiles(smiles))
                else:
                    smiles_label = str(smiles_index)
                write_structure(
                    os.path.join(output_dir, smiles_label, str(i)),
                    struct, diene_inds,
                    energy=engs[i],
                    smiles=smiles
                )

            final_smiles.append(smiles)
            final_structures.append(struct)
            energies.append(engs[i])
            indices.append(diene_inds)

    return final_smiles, final_structures, energies, indices

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
        preoptimize=False,
        optimizer_settings=None,
        smiles_hash_generator=None,
        output_dir=None,
        parallelizer=None,
        batch_size=50
):
    base_iterator = enumerate(
        product_smiles_iterator(
            templates, diene_atoms, group_map,
            diene_template=diene_template
        )
    )
    if parallelizer is None:
        return _generate_products_and_optimize(
            base_iterator,
            conf_gen_options=conf_gen_options,
            take_unique=take_unique,
            num_structs=num_structs,
            calc=calc,
            evaluate_energy=evaluate_energy,
            preoptimize=preoptimize,
            optimizer_settings=optimizer_settings,
            smiles_hash_generator=smiles_hash_generator,
            output_dir=output_dir
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
                    preoptimize=preoptimize,
                    optimizer_settings=optimizer_settings,
                    smiles_hash_generator=smiles_hash_generator,
                    output_dir=output_dir
                ),
                batches
            ):
                final_smiles.extend(f_smiles)
                final_structures.extend(f_struct)
                energies.extend(f_eng)
                indices.extend(f_ind)

        return final_smiles, final_structures, energies, indices

def old():
    R_list = ["O=S(=O)(C2=CC=CC=C2)", "NC", "NC(=O)", "FC(C=C3)=CC=C3"]
    X_list = ["S(=O)(C4=CC=CC=C4)=O", "CN", "C(=O)N", "C5=CC=C(F)C=C5"]
    R_prime_list = ["S(=O)(C6=CC=CC=C6)=O", "CN", "C(=O)N", "C7=CC=C(F)C=C7"]
    R_heavy_atom_indices = [9, 2, 3, 7]


    template_1 = "[R][C@H]1N=C[C@@H]([X])[C@@H]([R'])C1"
    template_2 = "[R][C@@H]1N=C[C@H]([X])[C@H]([R'])C1"
    template_3 = "[R][C@@H]1N=C[C@H]([X])[C@@H]([R'])C1"
    template_4 = "[R][C@H]1N=C[C@@H]([X])[C@H]([R'])C1"
    template_5 = "[R][C@H]1N=C[C@@H]([X])C[C@H]1([R'])"
    template_6 = "[R][C@@H]1N=C[C@H]([X])C[C@@H]1([R'])"
    template_7 = "[R][C@@H]1N=C[C@H]([X])C[C@H]1([R'])"
    template_8 = "[R][C@H]1N=C[C@@H]([X])C[C@@H]1([R'])"


    # Generate combinations of R, X, and R'
    combinations = list(itertools.product(R_list, X_list, R_prime_list ))
    atom_indices = list(itertools.product(R_heavy_atom_indices, R_heavy_atom_indices, R_heavy_atom_indices))

    All_smiles = []
    Indices = []
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_1, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + atom_indices[idx][2] + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_2, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + atom_indices[idx][2] + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_3, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + atom_indices[idx][2] + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_4, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + atom_indices[idx][2] + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_5, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_6, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_7, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])
    for idx, (R, X, R_prime) in enumerate(combinations):
        smiles = generate_smiles(template_8, R, X, R_prime)
        idx1, idx2 = atom_indices[idx][0] + 1, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1 + 1
        idx3, idx4 = atom_indices[idx][0] + 4, atom_indices[idx][0] + 4 + atom_indices[idx][1] + 1
        All_smiles.append(smiles)
        Indices.append([[idx1,idx2],[idx3,idx4]])

    Canonical_smiles = []
    for i in All_smiles:
        Canonical_smiles.append(Chem.CanonSmiles(i))
    uniq_smiles, uniq_indices = np.unique(Canonical_smiles,return_index=True)

    f = open('R8_smiles.txt', 'w')
    for i in range(len(uniq_indices)):
        tmp_smiles = All_smiles[uniq_indices[i]]
        f.write(tmp_smiles + '\n')
    f.close()

    for i in range(len(uniq_indices)):
        tmp_smiles = All_smiles[uniq_indices[i]]
        mol = Chem.MolFromSmiles(tmp_smiles)
        mol = Chem.AddHs(mol)

        atoms = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol())
        Energy_all = []
        for j in gen_conformers(mol):
            conf = mol.GetConformer(j)
            coord = conf.GetPositions()
            conformer_tmp = Atoms(symbols=atoms, positions=coord)
            conformer_tmp.calc = calc
            energy = conformer_tmp.get_potential_energy()
            Energy_all.append(energy[0])

        Energy_all = np.array(Energy_all)
        Energy_relative = Energy_all - min(Energy_all)
        top_indices = np.argpartition(Energy_relative, 10)[:10]

        try:
            os.mkdir('Reaction8/'+str(i))
        except:
            pass

        for j in top_indices:
            conf = mol.GetConformer(int(j))
            coord = conf.GetPositions()
            try:
                os.mkdir('Reaction8/'+str(i)+'/'+str(j))
            except:
                pass

            write('Reaction8/'+str(i)+'/'+str(j)+'/conf.xyz', conformer_tmp, format='xyz')

            f = open('Reaction8/'+str(i)+'/'+str(j)+'/isomer.txt','w')
            f.write('BREAK ' + str(Indices[uniq_indices[i]][0][0]) + " " + str(Indices[uniq_indices[i]][0][1]) + '\n')
            f.write('BREAK ' + str(Indices[uniq_indices[i]][1][0]) + " " + str(Indices[uniq_indices[i]][1][1]) + '\n')
            f.close()


def test_main():
    dienes = [
        ["C@H", "N", "C@@H", "C@@H"],
        ["C@@H", "N", "C@H", "C@H"],
        ["C@@H", "N", "C@H", "C@@H"],
        ["C@H", "N", "C@@H", "C@H"]
    ]
    base_templates = [
        "[R][diene]([X])[C@@H:3]([R'])[C:4]1",
        "[R][diene]([X])[C:3][C@H:4]1([R'])"
    ]
    replacements = {
        'R': ["O=S(=O)(C2=CC=CC=C2)", "NC", "NC(=O)", "FC(C=C3)=CC=C3"],
        'X': ["S(=O)(C4=CC=CC=C4)=O", "CN", "C(=O)N", "C5=CC=C(F)C=C5"],
        "R'": ["S(=O)(C6=CC=CC=C6)=O", "CN", "C(=O)N", "C7=CC=C(F)C=C7"]
    }

    dat = generate_products_and_optimize(
        base_templates[:1],
        dienes,
        {
            k:v
            for k,v in replacements.items()
        },
        evaluate_energy=False,
        output_dir=os.path.expanduser('~/Desktop/test_smi'),
        parallelizer=True
    )

