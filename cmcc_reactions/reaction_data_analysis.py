
import collections
import numpy as np
from . import reaction_data_schema as schema

__all__ = [
    "enumerate_distortions",
    "load_distortion",
    "enumerate_result_set",
    "get_reactant_and_ts"
]

ignored_paths = {'atoms'}
def enumerate_distortions(distortion_data:dict, filter=None, mode='dfs', max_depth=5):
    #DFS walk of the tree
    queue = collections.deque([[{'path':[]}, distortion_data]])
    if mode.lower() == 'dfs':
        pop = queue.pop
        append = queue.append
    elif mode.lower() == 'bfs':
        pop = queue.pop
        append = queue.appendleft
    else:
        raise ValueError(f"bad mode {mode}")
    while queue:
        meta, tree = pop()
        if max_depth is not None and len(meta['path']) >= max_depth:
            yield {'base_data':dict(meta), 'leaf_node':tree}
        if 'results' in tree:
            yield dict(meta, **tree)
        elif not isinstance(tree, dict):
            for k,v in enumerate(tree):
                new_meta = collections.ChainMap({'path':meta['path'] + [k]}, meta)
                if filter is not None and not filter(new_meta):
                    continue
                append([new_meta, v])
        else:
            if 'atoms' in tree:
                meta = collections.ChainMap({'atoms':tree['atoms']}, meta)
            for k,v in tree.items():
                if k in ignored_paths: continue
                new_meta = collections.ChainMap({'path':meta['path'] + [k]}, meta)
                if filter is not None and not filter(new_meta):
                    continue
                append([new_meta, v])

def load_distortion(distortion_data:dict, *path, atoms=None):
    data = distortion_data
    if atoms is None and isinstance(data, dict) and 'atoms' in data:
        atoms = data['atoms']
    for p in path:
        data = data[p]
        if atoms is None and isinstance(data, dict) and 'atoms' in data:
            atoms = data['atoms']
    if atoms is not None:
        data = dict({'atoms':atoms}, **data)
    return data

def enumerate_result_set(results:dict):
    if 'coordinates' in results:
        res2 = results.copy()
        atoms = res2.pop('atoms')
        coords = res2.pop('coordinates')
        yield (atoms, coords), res2
    else:
        atoms = results['atoms']
        zip_keys = [
            k for k,v in results.items()
            if k not in {'results', 'atoms', 'spec', 'path'} and isinstance(v, (list, dict, np.ndarray))
        ]
        rem_dict = {
            k:v
            for k,v in results.items()
            if k not in {'results', 'atoms', 'path'} and k not in zip_keys
        }
        for prod in zip(results['results'], *(results[k] for k in zip_keys)):
            res2 = prod[0].copy()
            # we descend down the tree until we hit a proper structure dict with coordinates
            queue = collections.deque([[{'path':[]}, res2]])
            while queue:
                meta, tree = queue.pop()
                if 'coordinates' in tree:
                    tree = tree.copy()
                    tree.update(meta)
                    tree.update(rem_dict)
                    tree.update(zip(zip_keys, prod[1:]))
                    coords = tree.pop('coordinates')
                    yield (atoms, coords), tree
                elif not isinstance(tree, dict):
                    for k, v in enumerate(tree):
                        new_meta = collections.ChainMap({'path': meta['path'] + [k]}, meta)
                        queue.append([new_meta, v])
                else:
                    for k, v in tree.items():
                        new_meta = collections.ChainMap({'path': meta['path'] + [k]}, meta)
                        queue.append([new_meta, v])

def get_reactant_and_ts(distortion_data):
    reactant = None
    transition_state = None
    for (atoms, coords), meta in enumerate_result_set(distortion_data):
        if 'transition_state' in meta['path']:
            transition_state = (atoms, coords), meta
        elif 'reactant' in meta['path']:
            reactant = (atoms, coords), meta
        if transition_state is not None and reactant is not None:
            return reactant, transition_state
    else:
        if transition_state is None and reactant is None:
            raise ValueError("couldn't load reactant and transition state")
        elif transition_state is None:
            raise ValueError("couldn't load transition state")
        elif reactant is None:
            raise ValueError("couldn't load reactant")

def build_mol(structure_data, **opts):
    from Psience.Molecools import Molecule
    from McUtils.Data import UnitsData

    (atoms, coords), meta = structure_data
    return Molecule(atoms, np.array(coords) * UnitsData.convert("Angstroms", "BohrRadius"),
                    **dict(meta, **opts))

def load_all_structs(distortion_data:dict):
    mols = {}
    for k in enumerate_distortions(distortion_data):
        if k['results'] is not None:
            for (atoms, coords), meta in enumerate_result_set(k):
                full_path = tuple(k['path'] + meta['path'])
                meta['path'] = full_path
                mols[full_path] = (atoms, coords), meta
    return mols