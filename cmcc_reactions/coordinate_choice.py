
import numpy as np, itertools
import McUtils.Numputils as nput
import McUtils.Iterators as itut
import McUtils.Coordinerds as coordops
from McUtils.Graphs import EdgeGraph
from Psience.Molecools import Molecule

def resolve_dropped_bonds(sel):
    if len(sel) == 2:
        return [sel]
    elif len(sel) == 3:
        return [(sel[1], sel[2])]
    elif len(sel) == 4:
        return [(sel[2], sel[3])] #TODO: is this correct...?
    else:
        raise ValueError(f"can't resolve bonds to break for {sel}")

def get_bond_fragment_atoms(full_atoms, edges, dropped_bonds):

    graph = EdgeGraph(full_atoms, edges)
    base_frags = graph.get_fragments()

    sym_bonds = {
        (i,j) for i,j in dropped_bonds
    } | {
        (j,i) for i,j in dropped_bonds
    }

    new_edges = [
        (i,j)
        for i,j in edges
        if (i,j) not in sym_bonds
    ]

    clip_graph = EdgeGraph(full_atoms, new_edges)
    clip_frags = clip_graph.get_fragments()

    if (
            len(clip_frags) == len(base_frags)
            and all(any(len(x) == len(y) and np.all(x==y) for y in base_frags) for x in clip_frags)
    ):
        return dropped_bonds, (base_frags, clip_frags), None

    bond_frag_atoms = {}
    for i,j in dropped_bonds:
        for g in clip_frags:
            if i in g:
                bond_frag_atoms[i] = g
            if j in g:
                bond_frag_atoms[j] = g
        bond_frag_atoms[i] = bond_frag_atoms.get(i, [])
        bond_frag_atoms[j] = bond_frag_atoms.get(j, [])

    return dropped_bonds, (base_frags, clip_frags), bond_frag_atoms

def get_carried_atoms(edges, ats, return_bond=False):
    full_atoms = np.unique(np.concatenate(edges))
    dropped_bonds, _, bond_frag_atoms = get_bond_fragment_atoms(full_atoms, edges, resolve_dropped_bonds(ats))

    at1, at2 = dropped_bonds[0]
    if bond_frag_atoms is None or sorted(bond_frag_atoms[at1]) == sorted(bond_frag_atoms[at2]):
        raise ValueError(f"bond graph still connected after breaking bonds {dropped_bonds}")

    carried_atoms = bond_frag_atoms[at2]
    fixed_atoms = bond_frag_atoms[at1]

    if return_bond:
        return (carried_atoms, fixed_atoms), dropped_bonds[0]
    else:
        return carried_atoms, fixed_atoms

def _get_mostly_fixed_coordinate_system(mol:Molecule,
                                       target_coords,
                                       fragment_connection=True,
                                       keep_any=False):

    base_sys = mol.get_bond_graph_internals()
    frags = mol.fragment_indices
    edges = [b[:2] for b in mol.bonds]
    # connect fragments through the target coords
    if len(frags) > 1:
        if len(frags) > 2:
            raise NotImplementedError(f"3 or more fragments not handled (got {len(frags)})")
        if fragment_connection:
            if fragment_connection is True:
                if len(target_coords[0]) == 4:
                    base_ats = [target_coords[0][i] for i in [1, 2, 0]]
                elif len(target_coords[0]) == 3:
                    base_ats = [target_coords[0][i] for i in [1, 0, 2]]
                else:
                    base_ats = np.concatenate(target_coords)[:3]
                other_frag = [f for f in frags if base_ats[0] not in f][0]
                dists = np.linalg.norm(mol.coords[other_frag,] - mol.coords[base_ats[0]][np.newaxis], axis=1)
                a_pos = np.argmin(dists)
                a = other_frag[a_pos]
                for i,j in edges:
                    if a == i:
                        b = j
                        break
                    elif a == j:
                        b = i
                        break
                else:
                    dists = np.linalg.norm(mol.coords[other_frag,] - mol.coords[a][np.newaxis], axis=1)
                    dists[dists == 0] = np.max(dists) + 1
                    b = other_frag[np.argmin(dists)]
                j, k, l = base_ats
                fragment_connection = [
                    (a, j),
                    (a, j, k),
                    (a, j, k, l),
                    (b, j),
                    (b, a, j),
                    (b, a, j, k),
                ]
                # if len(other_frag) > 2:
                #     for i,j in edges:
                #         if b == i and a != j:
                #             c = j
                #             break
                #         elif b == j and a != i:
                #             c = i
                #             break
                #     else:
                #         dists = np.linalg.norm(mol.coords[other_frag,] - mol.coords[b][np.newaxis], axis=1)
                #         dists[dists == 0] = np.max(dists) + 1
                #         dists[a_pos] = np.max(dists) + 1
                #         c = other_frag[np.argmin(dists)]
                #     fragment_connection.append((c, b, a, j))

    constraints = []
    keep_coords = []
    test_vecs = []
    for tc in target_coords:
        carried_atoms, fixed_atoms = get_carried_atoms(edges, tc)
        for coord in base_sys:
            if keep_any:
                if any(i in carried_atoms for i in coord):
                    keep_coords.append(coord)
            else:
                if all(i in carried_atoms for i in coord):
                    keep_coords.append(coord)
        # constraints.extend(itertools.combinations(fixed_atoms, 2))
        if len(frags) > 1:
            for a in fixed_atoms:
                other_frag = [f for f in frags if a not in f][0]
                dists = np.linalg.norm(mol.coords[other_frag,] - mol.coords[a][np.newaxis], axis=1)
                _, b_pos = nput.partial_sort(dists, 3, return_order=True)
                constraints.extend((a, other_frag[b]) for b in b_pos)
        vec = nput.internal_coordinate_tensors(mol.coords, [tc], order=1, fixed_atoms=fixed_atoms)[1]
        test_vecs.append(vec)

    keep_hash = set(keep_coords)
    drop_coords = [c for c in base_sys if c not in keep_hash]

    base_sys = list(itut.delete_duplicates(
        (tuple(c) for c in
         itertools.chain(target_coords,
                        keep_coords,
                        constraints,
                        [] if not fragment_connection else fragment_connection,
                        drop_coords
                        )),
        key=coordops.canonicalize_internal
    ))

    target_coords = [tuple(c) for c in target_coords]
    cache = set(coordops.canonicalize_internal(c) for c in target_coords)
    keep_coords, constraints, fragment_connection, drop_coords = [
        list(itut.delete_duplicates((tuple(c) for c in subset),
                                    key=coordops.canonicalize_internal,
                                    cache=cache))
        for subset in [keep_coords, constraints, fragment_connection, base_sys]
    ]

    # targ_vecs = nput.internal_coordinate_tensors(mol.coords, base_sys, order=1)[1]
    # tf = nput.maximum_similarity_transformation(targ_vecs, np.concatenate(test_vecs, axis=1), apply_transformation=False)
    # max_contrib = np.max(tf**2, axis=1)
    # optimals = np.argsort(max_contrib)


    return (target_coords, keep_coords, constraints, fragment_connection, drop_coords), test_vecs

def _get_atom_diplacement_coordinate_system(mol:Molecule,
                                           target_coords,
                                           include_intra_contraints=True,
                                           return_components=False,
                                           prune=True):
    carried_coords = []
    orientation_coords = []
    constraints = []
    key_constraints = []
    fragment_constraints = []
    key_orientation = []

    frags = mol.fragment_indices
    edges = [b[:2] for b in mol.bonds]
    _prev_frag_idx = None
    for tc in target_coords:
        carried_atoms, fixed_atoms = get_carried_atoms(edges, tc)
        if len(tc) > 2:
            key_orientation.append(tc[1:])
            if len(tc) > 3:
                key_orientation.append(tc[2:])

        targ = tc[-1]
        if len(frags) > 1:
            frag_idx = [i for i,f in enumerate(frags) if targ in f][0]
            if _prev_frag_idx is not None and frag_idx != _prev_frag_idx:
                raise ValueError(f"coords {target_coords} are on different fragments, no consistent internal set can be generated")
        else:
            frag_idx = None

        fixed_set = set(fixed_atoms)
        invovled_fixed = fixed_set & set(tc)
        uninvovled_fixed = fixed_set - set(tc)
        orientation_coords.extend(
            x for x in
            mol.get_bond_graph_internals(fragment=frag_idx, include_stretches=False, include_bends=False, include_dihedrals=True)
            if targ in x and len(set(x) & uninvovled_fixed) == 0
        )

        carried_set = set(carried_atoms)
        carried_coords.extend(
            x for x in
            # coordops.extract_zmatrix_internals(mol.get_bond_zmatrix(for_fragment=frag_idx))
            mol.get_bond_graph_internals(fragment=frag_idx)
            if len(set(x) & carried_set) == len(x)
        )

        for fi in np.setdiff1d(np.arange(len(frags)), [frag_idx]):
            if _prev_frag_idx is None:
                fragment_constraints.extend(
                    mol.get_bond_graph_internals(fragment=fi, include_bends=False, include_dihedrals=False)
                )

            key_constraints.extend(
                (i, j) for i in frags[frag_idx] for j in frags[fi]
               if i in invovled_fixed
            )
            active_set = carried_set | invovled_fixed
            constraints.extend(
                (i, j) for i in frags[frag_idx] for j in frags[fi]
               if i not in active_set
            )

        if include_intra_contraints:
            constraints.extend(
                (i, j) for i,j in itertools.combinations(fixed_atoms, 2)
            )

    bits = (
        list(target_coords),
        key_orientation, carried_coords, key_constraints,
        orientation_coords, fragment_constraints, constraints
    )
    const_sys = sum(bits, [])
    if prune:
        const_sys = mol.prune_internals(const_sys)

    if return_components:
        return const_sys, bits
    else:
        return const_sys

def get_mostly_fixed_coordinate_system(mol:Molecule,
                                       target_coord):
    _, bond = get_carried_atoms([b[:2] for b in mol.bonds], target_coord, return_bond=True)

    mol_mod = mol.break_bonds([bond])
    frags = sorted(
        mol_mod.fragment_indices,
        key=lambda f:(1000
                        if bond[1] in f else
                      100
                        if bond[0] in f else
                      0) - len(f)
    )
    zm = mol_mod.get_bond_zmatrix(
        fragments=frags,
        attachment_points={bond[1]:bond[0]}
    )
    main_atom = bond[1]
    specs = sorted(coordops.extract_zmatrix_internals(zm),
                   key=lambda c:-(
                           (100 if c[0] == main_atom or c[-1] == main_atom else 0)
                           + (50 if len(c) == len(target_coord) else 0)
                           + len(np.intersect1d(c, target_coord))
                   ))
    return specs