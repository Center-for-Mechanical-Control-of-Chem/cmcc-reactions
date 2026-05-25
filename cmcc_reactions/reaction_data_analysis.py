
import collections
import os.path
import numpy as np
import glob
import itertools

from . import reaction_data_schema as schema
from . import utils
from . import generate_reaction_products as gen_prods
from . import trajectory_tools as trajt
from . import pipeline

import McUtils.Devutils as dev
from McUtils.Data import UnitsData, BondData
import McUtils.Numputils as nput
import McUtils.Plots as plt
import McUtils.Iterators as itut
from Psience.Molecools import Molecule

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

class ForceModifiedReactionAnalyzer:
    def __init__(self,
                 atoms,
                 reactant_geom, transition_state_geom,
                 force_modified_reactant_geom, force_modified_transition_state_geom,
                 reactant_energy, transition_state_energy,
                 force_modified_reactant_energy, force_modified_transition_state_energy,
                 ):
        self.reactant_energy = reactant_energy
        self.transition_state_energy = transition_state_energy
        self.force_modified_reactant_energy = force_modified_reactant_energy
        self.force_modified_transition_state_energy = force_modified_transition_state_energy
        self.reactant = Molecule(atoms, reactant_geom)
        self.force_modified_reactant = Molecule(atoms, force_modified_reactant_geom)
        self.transition_state = Molecule(atoms, transition_state_geom)
        self.force_modified_transition_state = Molecule(atoms, force_modified_transition_state_geom)

    def animate_reactant_distortion(self, **opts):
        return self.reactant.plot([
            self.reactant.coords,
            self.force_modified_reactant.coords
        ], **opts)

    def animate_ts_distortion(self, **opts):
        return self.transition_state.plot([
            self.transition_state.coords,
            self.force_modified_transition_state.coords
        ], **opts)

class BarrierHeightDataset:
    def __init__(self,
                 reactant_energies,
                 force_modified_reactant_energies,
                 transition_state_energies,
                 force_modified_transition_state_energies,
                 *,
                 atoms=None,
                 reactant_geometries=None,
                 force_modified_reactant_geometries=None,
                 transition_state_geometries=None,
                 force_modified_transition_state_geometries=None,
                 force_magnitudes=None,
                 force_vectors=None,
                 dataset=None,
                 **meta_fields):
        self.reactant_energies = np.asanyarray(reactant_energies)
        self.fm_reactant_energies = np.asanyarray(force_modified_reactant_energies)
        self.transition_state_energies = np.asanyarray(transition_state_energies)
        self.fm_transition_state_energies = np.asanyarray(force_modified_transition_state_energies)
        self.reactant_geometries = reactant_geometries
        self.force_modified_reactant_geometries = force_modified_reactant_geometries
        self.transition_state_geometries = transition_state_geometries
        self.force_modified_transition_state_geometries = force_modified_transition_state_geometries
        if force_magnitudes is not None:
            force_magnitudes = np.asanyarray(force_magnitudes)
        self.force_magnitudes = force_magnitudes
        self.force_vectors = force_vectors
        self.atoms = atoms
        self.meta_fields = {
            k: np.asanyarray(v) if nput.is_numeric_array_like(v) else v
            for k, v in meta_fields.items()
        }

        self.barriers = self.transition_state_energies - self.reactant_energies
        self.delta_r = self.fm_reactant_energies - self.reactant_energies
        self.delta_t = self.fm_transition_state_energies - self.transition_state_energies
        self.fm_barriers = self.fm_transition_state_energies - self.fm_reactant_energies
        self.deltas = self.fm_barriers - self.barriers
        self.dataset = dataset
        for k, v in meta_fields.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{type(self).__name__}<{len(self)}>"

    def __len__(self):
        return len(self.barriers)
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.load_opt_res(index=i)

    def get_tree_data(self, index):
        if self.dataset is None:
            raise ValueError("No dataset specified")
        id = self.meta_fields['data_ids'][index]
        t = self.dataset
        if nput.is_int(id) or isinstance(id, str): id = [id]
        for tt in id:
            t = t[tt]
        return t
    def load_trajectory(self, index, **etc):
        return trajt.DielsAlderReactionTrajectory.from_trajectory_data(
            self.get_tree_data(index),
            **etc
        )
    def load_fmra(self, index, **etc):
        return ForceModifiedReactionAnalyzer(
            self.atoms[index],
            self.reactant_geometries[index], self.transition_state_geometries[index],
            self.force_modified_reactant_geometries[index], self.force_modified_transition_state_geometries[index],
            self.reactant_energies[index], self.transition_state_energies[index],
            self.fm_reactant_energies[index], self.fm_transition_state_energies[index],
            **etc
        )
    def load_opt_res(self, index, force_indices=None, **etc):
        if force_indices is None:
            force_indices = [self.fmrd_ids[index]]
        res0 = pipeline.OptimizedForceResults.from_intermediate_data(
            self.get_tree_data(index),
            **etc
        )
        if len(force_indices) > 0:
            res0.fmrds = [res0.fmrds[index] for index in force_indices]
        return res0

    # def __getattr__(self, name):
    #     return self.meta_fields[name]

    def filter_by_mask(self, mask):
        mi = np.where(mask)[0]
        opts = {
            k: [fms[i] for i in mi] if fms is not None else None
            for k, fms in {
                'atoms': self.atoms,
                'force_magnitudes': self.force_magnitudes,
                'force_vectors': self.force_vectors,
                'reactant_geometries': self.reactant_geometries,
                'transition_state_geometries': self.transition_state_geometries,
                'force_modified_reactant_geometries': self.force_modified_reactant_geometries,
                'force_modified_transition_state_geometries': self.force_modified_transition_state_geometries
            }.items()
        } | {
                k:[v[i] for i in mi]
                for k, v in self.meta_fields.items()
            }

        return type(self)(
            self.reactant_energies[mask],
            self.fm_reactant_energies[mask],
            self.transition_state_energies[mask],
            self.fm_transition_state_energies[mask],
            dataset=self.dataset,
            **opts
        )

    def get_filter_data(self, energy_units="Kilocalories/Mole", force_units="Picojoules/Meters"):
        energy_props = {
            'reactant_energies': self.reactant_energies,
            'force_modified_reactant_energies': self.fm_reactant_energies,
            'transition_state_energies': self.transition_state_energies,
            'force_modified_transition_state_energies': self.fm_transition_state_energies,
            'barrier': self.barriers,
            'force_modified_barrier': self.fm_barriers,
            'delta': self.deltas,
            'delta_reactant': self.delta_r,
            'delta_transition_state': self.delta_t,
        }
        energy_props = {
            k:e * UnitsData.convert("Hartrees", energy_units)
            for k,e in energy_props.items()
        }
        force_props = {
            'force_magnitudes': self.force_magnitudes
        }
        force_props = {
            k: e * UnitsData.convert("Hartrees/BohrRadius", force_units)
            for k, e in force_props.items()
        }
        return energy_props | force_props | {
            'atoms':self.atoms,
            'transition_state_geometries':self.transition_state_geometries,
            'force_modified_transition_state_geometries':self.force_modified_transition_state_geometries,
            'reactant_geometries':self.reactant_geometries,
            'force_modified_reactant_geometries':self.force_modified_reactant_geometries,
            'force_vectors': self.force_vectors
        } | self.meta_fields
    def get_filter_mask(self,
                        filter_map,
                        energy_units="Kilocalories/Mole",
                        force_units="Picojoules/Meters"
                        ):
        filter_data = self.get_filter_data(
            energy_units=energy_units,
            force_units=force_units
        )
        mask = np.arange(len(self.reactant_energies))
        if callable(filter_map): filter_map = [filter_map]
        for filter_function in filter_map:
            new = filter_function(filter_data)
            sub = np.where(new)[0]
            filter_data = {
                k: (
                    v[sub,]
                        if isinstance(v, np.ndarray) else
                    [v[i] for i in sub]
                        if v is not None else
                    None
                )
                for k,v in filter_data.items()
            }
            mask = mask[new]
        _ = np.full(len(self.reactant_energies), False)
        _[mask] = True
        return _, filter_data

    @classmethod
    def aggregate_mask_inds(cls, keys):
        if nput.is_atomic(keys[0]):
            keys = [keys]
        id_groups = [nput.group_indices(k)[0] for k in keys]
        for id_idx_groups in itertools.product(*[zip(*gg) for gg in id_groups]):
            ids = tuple(i.tolist() for i, g in id_idx_groups)
            if len(ids) == 1: ids = ids[0]
            g = id_idx_groups[0][1]
            for _, g2 in id_idx_groups[1:]:
                g = np.intersect1d(g, g2)
            yield ids, g
    @classmethod
    def aggregate_mask_values(cls, values, keys):
        if isinstance(values, dict):
            values = {
                k: np.asanyarray(v) if nput.is_array_like(v) else v
                for k, v in values.items()
            }
        elif nput.is_array_like(values):
            values = np.asanyarray(values)
        for ids, g in cls.aggregate_mask_inds(keys):
            if isinstance(values, dict):
                subvals = {
                    k: v[g,] if isinstance(values, np.ndarray) else [v[i] for i in g]
                    for k, v in values.items()
                }
            else:
                subvals = values[g,] if isinstance(values, np.ndarray) else [values[i] for i in g]
            yield ids, g, subvals

    @classmethod
    def aggregate_by_groups(cls, values, keys):
        res = {}
        for ids, g, subvals in cls.aggregate_mask_values(values, keys):
            res[ids] = subvals
        return res

    @classmethod
    def group_mask(cls, values, keys, filter, mode='any'):
        mask = np.full(len(keys), False if mode == 'any' else True)
        for ids, g, subvals in cls.aggregate_mask_values(values, keys):
            submask = filter(ids, subvals)
            mask[g[submask],] = True if mode == 'any' else False
        return mask

    @classmethod
    def mask_first(cls, data, n):
        mask = np.full(len(data), False)
        mask[:n] = True
        return mask
    @classmethod
    def mask_last(cls, data, n):
        mask = np.full(len(data), False)
        mask[-n:] = True
        return mask
    @classmethod
    def last_few(cls, keys, n):
        return cls.group_mask(np.zeros(len(keys)), keys,
                              lambda d:cls.mask_last(d, n))
    @classmethod
    def first_few(cls, keys, n):
        return cls.group_mask(np.zeros(len(keys)), keys,
                              lambda d:cls.mask_first(d, n))

    @classmethod
    def match_sets(cls, value_arrays, matches):
        value_arrays = [np.asanyarray(v) if nput.is_array_like(v) else v for v in value_arrays]
        match_arrays = itut.transpose(matches)
        match_unique = [np.unique(m).tolist() for m in match_arrays]
        array_matches_unique = [
            {
                m: a == m if isinstance(a, np.ndarray) else np.array([aa == m for aa in a])
                for m in match_list
            }
            for a, match_list in zip(value_arrays, match_unique)
        ]
        mask = np.full(len(value_arrays[0]), False)
        for match in matches:
            submask = array_matches_unique[0][match[0]]
            for a, m in zip(array_matches_unique[1:], match[1:]):
                submask = submask & a[m]
            mask = mask | submask
        return mask

    def filter_by_props(self,
                        filter_map,
                        energy_units="Kilocalories/Mole",
                        force_units="Picojoules/Meters"
                        ):
        mask, _ = self.get_filter_mask(filter_map, energy_units=energy_units, force_units=force_units)
        return self.filter_by_mask(mask)

    def aggregate_by_props(self, value_keys, aggregation_keys,
                           energy_units="Kilocalories/Mole",
                           force_units="Picojoules/Meters"
                           ):
        filter_data = self.get_filter_data(
            energy_units=energy_units,
            force_units=force_units
        )
        if isinstance(aggregation_keys, str): aggregation_keys = [aggregation_keys]
        if isinstance(value_keys, str): value_keys = [value_keys]
        return self.aggregate_by_groups(tuple(filter_data[v] for v in value_keys),
                                        tuple(filter_data[k] for k in aggregation_keys))

    def plot(self, color=None, force_units="Picojoules/Meters", **etc):
        if color is None and self.force_magnitudes is not None:
            force_units = force_units.replace("newtons", "Newtons").replace("Newtons", "joules/Meters")
            color = self.force_magnitudes * UnitsData.convert("Hartrees/BohrRadius", force_units)

        fig = plt.ScatterPlot(self.barriers * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
                              self.deltas * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
                              **(
                                      dict(
                                      color=color,
                                      axes_labels=[r"$E_a^\text{solv}$ (kcal mol$^{-1}$)",
                                                   r"$\Delta E_a^\text{mech}$ (kcal mol$^{-1}$)"]
                                  ) | etc
                              )
                              )
        fig = plt.Plot(fig.plot_range[0], [0, 0], figure=fig, linestyle='dashed')
        return fig

    @classmethod
    def from_dataset_loader(cls,
                            loader,
                            field_map=None,
                            filter=None,
                            discarded_keys=None,
                            annotation_generator=None,
                            **etc):

        if field_map is None:
            field_map = {
                'vectors':{
                    'force_modified_transition_state_energies':'force_modified_transition_state_energies',
                    'force_modified_reactant_energies':'force_modified_reactant_energies',
                    'force_magnitudes':'force_magnitudes',
                    'force_vectors':'force_vectors',
                    'force_modified_transition_state_geometries':'force_modified_transition_state_geometries',
                    'force_modified_reactant_geometries':'force_modified_reactant_geometries',
                },
                'scalars':{
                    'atoms':'atoms',
                    'reactant_energies':'reactant_energy',
                    'transition_state_energies':'transition_state_energy',
                    'reactant_geometries':'reactant_geometry',
                    'transition_state_geometries':'transition_state_geometry'
                }
            }

        results = {}
        for f in field_map['vectors']:
            results[f] = []
        for f in field_map['scalars']:
            results[f] = []

        data_ids = []
        fmrd_ids = []
        direction_ids = []
        magnitude_ids = []
        for id,data in loader:
            check = field_map['vectors']['force_modified_transition_state_energies']
            fmres = data.get(check)
            if fmres is None: continue
            if filter is not None:
                test = filter(data)
                if not test: continue
            if discarded_keys is not None:
                needs_copy = True
                for k in discarded_keys:
                    if needs_copy and k in data:
                        data = data.copy()
                        needs_copy = False
                    data.pop(k)

            nterms = len(fmres)
            for k,f in field_map['vectors'].items():
                results[k].extend(data[f])
            for k,f in field_map['scalars'].items():
                results[k].extend([data[f]] * nterms)

            data_ids.extend([id] * nterms)
            group_values, group_indices = nput.group_by(np.arange(nterms), data['force_magnitudes'])[0]
            mag_ids = np.zeros(nterms, dtype=int)
            for i,f in enumerate(group_indices): mag_ids[f] = i
            magnitude_ids.extend(mag_ids)

            # directions swap whenever groups cycle
            d_ids = np.zeros(nterms, dtype=int)
            _, splits = nput.group_by(np.arange(nterms), mag_ids)[0]
            old = splits[0][1]
            i = 0
            for i,s in enumerate(splits[0][2:]):
                d_ids[old:s] = i + 1
                old = s
            d_ids[old:] = i + 1
            direction_ids.extend(d_ids)

            fmrd_ids.extend(np.arange(nterms))


            if annotation_generator is not None:
                annotation_data = annotation_generator(id, data)
                for k,v in annotation_data.items():
                    if dev.is_atomic(v) or len(v) < nterms:
                        v = [v] * nterms
                    if k not in results: results[k] = []
                    results[k].extend(v)


        reactant_energies = results.pop('reactant_energies')
        fm_reactant_energies = results.pop('force_modified_reactant_energies')
        transition_state_energies = results.pop('transition_state_energies')
        fm_transition_state_energies = results.pop('force_modified_transition_state_energies')
        return cls(
            reactant_energies,
            fm_reactant_energies,
            transition_state_energies,
            fm_transition_state_energies,
            data_ids=data_ids,
            direction_ids=direction_ids,
            fmrd_ids=fmrd_ids,
            magnitude_ids=magnitude_ids,
            **etc,
            **results
        )

    @classmethod
    def from_file_list(cls, files,
                       dataset=None,
                       discarded_keys=("initial_trajectory_hessians", "refined_trajectory_hessians"),
                       file_loader=None, **opts):
        if dataset is None:
            dataset = {}
        if file_loader is None:
            file_loader = dev.read_json
        def loader():
            for f in files:
                data = file_loader(f)
                if discarded_keys is not None:
                    needs_copy = True
                    for k in discarded_keys:
                        if needs_copy and k in data:
                            data = data.copy()
                            needs_copy = False
                        data.pop(k)
                dataset[f] = data
                yield f, dataset[f]
        return cls.from_dataset_loader(loader(), dataset=dataset, **opts)

    @classmethod
    def from_file_pattern(cls, top_dir, js_pattern='**/pipeline_data.json', recursive=True,
                          file_filter=None,
                          max_files=None,
                          **opts):
        files = glob.glob(os.path.join(top_dir, js_pattern), recursive=recursive)
        if file_filter is not None:
            files = file_filter(files)
        if max_files is not None:
            files = files[:max_files]
        return cls.from_file_list(
            files,
            **opts
        )

    @classmethod
    def from_tree(cls, tree, depth=1, **opts):
        def loader(tree, depth, prefix=None):
            for k, v in tree.items():
                if depth > 0:
                    if prefix is None:
                        yield from loader(v, depth-1, (k,))
                    else:
                        yield from loader(v, depth-1, prefix=prefix+(k,))
                else:
                    if prefix is None:
                        yield (k,), v
                    else:
                        yield prefix+(k,), v
        return cls.from_dataset_loader(
            loader(tree, depth),
            dataset=tree,
            **opts
        )
