
import collections
import functools
import os.path
import numpy as np
import glob
import itertools
import multiprocessing
from scipy.optimize import least_squares

import rdkit.Chem.AllChem as Chem
from McUtils.ExternalPrograms import RDMolecule

from . import utils
from . import trajectory_tools as trajt
from . import pipeline

import McUtils.Devutils as dev
from McUtils.Data import UnitsData, BondData
import McUtils.Numputils as nput
import McUtils.Plots as plt
import McUtils.Iterators as itut
import McUtils.Formatters as mfmt
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
                 energy_evaluator=None,
                 reactant_hessian=None,
                 transition_state_hessian=None,
                 force_vector=None,
                 force_magnitude=None,
                 force_units=None,
                 mass_weight=None,
                 internals=None,
                 optimizer_settings=None,
                 predistorted_data=None
                 ):
        self.reactant_energy = reactant_energy
        self.transition_state_energy = transition_state_energy
        self.force_modified_reactant_energy = force_modified_reactant_energy
        self.force_modified_transition_state_energy = force_modified_transition_state_energy
        self.reactant = Molecule(atoms, reactant_geom,
                                 energy_evaluator=energy_evaluator,
                                 potential_derivatives=[0, reactant_hessian] if reactant_hessian is not None else None,
                                 spin=1)
        self.force_modified_reactant = Molecule(atoms, force_modified_reactant_geom, energy_evaluator=energy_evaluator,
                                                spin=1)
        self.transition_state = Molecule(atoms, transition_state_geom,
                                         energy_evaluator=energy_evaluator,
                                         potential_derivatives=
                                         [0, np.array(transition_state_hessian)]
                                         if transition_state_hessian is not None else
                                         None,
                                         spin=1)
        self.force_modified_transition_state = Molecule(atoms, force_modified_transition_state_geom,
                                                        energy_evaluator=energy_evaluator,
                                                        spin=1)
        self.force_vector = force_vector
        self.force_magnitude = force_magnitude
        self.force_units = force_units
        self.mass_weight = mass_weight
        self.internals = internals
        self.optimizer_settings = optimizer_settings
        self.predistorted_data = predistorted_data

    @classmethod
    def from_data(cls, data):
        if not isinstance(data, dict):
            data = data._asdict()
        return cls(**data)

    def plot_lines(self,
                   bar_color=('gray', 'blue'),
                   bar_spacing=.2,
                   distance_metric=None,
                   bonds=((0, 2), (1, 3)),
                   baseline=None,
                   product=None,
                   product_energy=None,
                   figure=None,
                   **etc
                   ):
        distance_metric = trajt.resolve_distance_metric(distance_metric)
        coords1 = distance_metric(
            [self.reactant.coords, self.transition_state.coords],
            bonds
        ) * UnitsData.convert("BohrRadius", "Angstroms")
        engs1 = [
            self.reactant_energy,
            self.transition_state_energy
        ]
        if product is not None:
            pd = distance_metric(
                [product.coords],
                bonds
            ) * UnitsData.convert("BohrRadius", "Angstroms")
            coords1 = np.concatenate([coords1, pd], axis=0)
            engs1 = engs1 + [product_energy]

        coords2 = distance_metric(
            [self.force_modified_reactant.coords, self.force_modified_transition_state.coords],
            bonds
        ) * UnitsData.convert("BohrRadius", "Angstroms")
        engs2 = [
            self.force_modified_reactant_energy,
            self.force_modified_transition_state_energy
        ]
        if product is not None:
            coords2 = np.concatenate([coords2, pd], axis=0)
            engs2 = engs2 + [product_energy]

        if baseline is None:
            baseline = self.reactant_energy

        x, X = np.min([coords1, coords2]), np.max([coords1, coords2])
        x = x - (X - x) * .1
        X = X + (X - x) * .1
        figure = trajt.plot_reaction_lines(
            coords1,
            engs1,
            connect=True,
            ticks=False,
            baseline=baseline,
            color=bar_color[0],
            bar_spacing=bar_spacing,
            figure=figure,
            plot_range=[[x, X], None],
            **etc
        )

        trajt.plot_reaction_lines(
            coords2,
            engs2,
            connect=True,
            ticks=False,
            baseline=baseline,
            color=bar_color[1],
            bar_spacing=bar_spacing,
            figure=figure,
            **etc
        )

        return figure

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
    def get_reduced_dataset(self, reduce_fmrds=False):
        new_ds = {}
        groups, subinds = nput.group_indices(self.meta_fields['data_ids'])[0]
        if reduce_fmrds:
            raise NotImplementedError("reducing partial dataset by fmrd is tedious")
        for s in subinds:
            index = s[0]
            data = self.get_tree_data(index)
            id = self.meta_fields['data_ids'][index]
            subtree = new_ds
            for i in id[:-1]:
                if i not in subtree:
                    subtree[i] = {}
                subtree = subtree[i]
            subtree[id[-1]] = data
        return new_ds
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
        return self.filter_by_inds(mi)
    def get_data_fields(self):
        return {
            'reactant_energies': self.reactant_energies,
            'force_modified_reactant_energies': self.fm_reactant_energies,
            'transition_state_energies': self.transition_state_energies,
            'force_modified_transition_state_energies': self.fm_transition_state_energies,
            'atoms': self.atoms,
            'force_magnitudes': self.force_magnitudes,
            'force_vectors': self.force_vectors,
            'reactant_geometries': self.reactant_geometries,
            'transition_state_geometries': self.transition_state_geometries,
            'force_modified_reactant_geometries': self.force_modified_reactant_geometries,
            'force_modified_transition_state_geometries': self.force_modified_transition_state_geometries
        } | self.meta_fields
    def filter_by_inds(self, mi):
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
            self.reactant_energies[mi,],
            self.fm_reactant_energies[mi,],
            self.transition_state_energies[mi,],
            self.fm_transition_state_energies[mi,],
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
        smol = False
        if isinstance(values, dict):
            values = {
                k: np.asanyarray(v) if nput.is_array_like(v) else v
                for k, v in values.items()
            }
        elif nput.is_atomic(values[0]):
            smol = True
            if nput.is_array_like(values): values = np.asanyarray(values)
            values = [values]
        else:
            if nput.is_atomic(keys[0]):
                keys = [keys]
            smol = len(values) == len(keys[0])
            if smol:
                values = [values]
            values = [np.asanyarray(v) if nput.is_array_like(v) else v for v in values]

        for ids, g in cls.aggregate_mask_inds(keys):
            if isinstance(values, dict):
                subvals = {
                    k: v[g,] if isinstance(values, np.ndarray) else [v[i] for i in g]
                    for k, v in values.items()
                }
            else:
                subvals = [
                    v[g,] if isinstance(v, np.ndarray) else [v[i] for i in g]
                    for v in values
                ]
            if smol: subvals = subvals[0]
            yield ids, g, subvals

    @classmethod
    def aggregate_by_groups(cls, values, keys):
        res = {}
        for ids, g, subvals in cls.aggregate_mask_values(values, keys):
            res[ids] = subvals
        return res

    def add_aggregation_fields(self, field_generator=None, input='optimizer', pool=True, **opts):
        base_fields = self.get_data_fields()
        if field_generator is not None:
            subfields = self.dispatch_over_dataset(field_generator, pool=pool, input=input)
            new_agg = {
                f:[v]
                for f,v in subfields[0].items()
            }
            for s in subfields[1:]:
                for f,v in s.items():
                    new_agg[f].append(v)
            opts = opts | new_agg
        opts = base_fields | opts
        return type(self)(
            dataset=self.dataset,
            **(self.meta_fields | opts)
        )

    @staticmethod
    def _partial_iter(block, *, generator, **opts):
        return [generator(o, **opts) for o in block]

    def dispatch_over_dataset(self, generator, pool=True, input='optimizer', **opts):
        if self.dataset is None:
            raise ValueError("`dataset` must be supplied")
        managed_pool = False
        if pool is True:
            managed_pool = True
            pool = multiprocessing.Pool()
        elif nput.is_int(pool):
            managed_pool = True
            pool = multiprocessing.Pool(pool)
        use_opt = dev.str_is(input, 'optimizer')
        if pool:
            max_size = len(self)
            nproc = pool._processes
            block_size = max(max_size // nproc, 1)
            num_blocks = int(np.ceil(max_size / block_size))
            blocks = [
                [
                    self.load_opt_res(j) if use_opt else self.get_tree_data(j)
                    for j in range(block_size*i, min([(block_size*i+1), max_size]))
                ]
                for i in range(num_blocks)
            ]
            block_generator = functools.partial(self._partial_iter, generator=generator, **opts)
            if managed_pool:
                with pool:
                    res = pool.map(block_generator, blocks)
            else:
                res = pool.map(block_generator, blocks)

            return sum(res, [])
        else:
            return [
                generator(self.load_opt_res(j) if use_opt else self.get_tree_data(j), **opts)
                for j in range(len(self))
            ]

    @staticmethod
    def compute_default_descriptors(opt, compute_gamma=True, compute_sterics=True,
                                    compute_volumes=False,
                                    compute_rigid=False):
        res = {}
        use_internals = len(opt.fmrds[0].force_vector) < len(opt.optimizer.rs.atoms) * 3
        if compute_gamma:
            res['gammas'] = opt.optimizer.compute_gammas(
                0,
                displacements=opt.fmrds[0].force_vector[np.newaxis],
                use_internals=use_internals
            )
        if compute_sterics:
            res['sterics'] = opt.optimizer.get_distortion_steric_repulsions(
                0,
                displacements=opt.fmrds[0].force_vector[np.newaxis],
                use_internals=use_internals,
                disp_min=0,
                disp_max=1,
                density=2
            )[0]
        if compute_rigid:
            res['rigid'] = opt.optimizer.predicted_delta_from_forces(
                0,
                opt.fmrds[0].force_magnitude,
                units=opt.fmrds[0].force_units,
                displacements=opt.fmrds[0].force_vector[np.newaxis],
                use_internals=use_internals,
                disp_min=-5,
                disp_max=5,
                steps=25
                # density=2
            )
        if compute_volumes:
            res['volumes'] = opt.optimizer.get_distortion_volume_changes(
                displacements=opt.fmrds[0].force_vector[np.newaxis],
                use_internals=use_internals,
                disp_min=-1,
                disp_max=0,
                only_endpoints=True
            )
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

    def __getitem__(self, item):
        if callable(item):
            item = [item]
        if nput.is_int(item):
            return self.load_opt_res(item)
        elif isinstance(item, slice) or nput.is_int(item[0]) or (item[0] is True or item[0] is False):
            inds = np.arange(len(self.reactant_energies))
            if isinstance(item, slice) or item[0] is True or item[0] is False:
                inds = inds[item]
            else:
                inds = inds[item,]
            return self.filter_by_inds(inds)
        else:
            return self.filter_by_props(item)

    def sample(self, n):
        if len(self) < n:
            return self
        else:
            choice = np.random.choice(len(self), size=n, replace=False)
            return self.filter_by_inds(choice)

    def aggregate_by_props(self, value_keys, aggregation_keys,
                           energy_units="Kilocalories/Mole",
                           force_units="Picojoules/Meters"):
        filter_data = self.get_filter_data(
            energy_units=energy_units,
            force_units=force_units
        )
        if isinstance(aggregation_keys, str): aggregation_keys = [aggregation_keys]
        if isinstance(value_keys, str): value_keys = [value_keys]
        return self.aggregate_by_groups(tuple(filter_data[v] for v in value_keys),
                                        tuple(filter_data[k] for k in aggregation_keys))

    def group_by_props(self, aggregation_keys,
                           energy_units="Kilocalories/Mole",
                           force_units="Picojoules/Meters"):
        filter_data = self.get_filter_data(
            energy_units=energy_units,
            force_units=force_units
        )
        if isinstance(aggregation_keys, str): aggregation_keys = [aggregation_keys]
        mask_data = self.aggregate_by_groups(
            np.arange(len(self)),
            tuple(filter_data[k] for k in aggregation_keys)
        )
        return {k:self.filter_by_inds(i) for k,i in mask_data.items()}

    def plot(self, color=None, force_units="Picojoules/Meters", figure=None, plot_baseline=None, baseline=0,
             baseline_styles=None,
             direction_markers=None,
             **etc):
        if color is None and self.force_magnitudes is not None:
            force_units = force_units.replace("newtons", "Newtons").replace("Newtons", "joules/Meters")
            color = self.force_magnitudes * UnitsData.convert("Hartrees/BohrRadius", force_units)

        if plot_baseline is None:
            plot_baseline = figure is None

        barr = self.barriers * UnitsData.convert("Hartrees", "Kilocalories/Mole")
        delt = self.deltas * UnitsData.convert("Hartrees", "Kilocalories/Mole")
        labs = [r"$E_a^\text{solv}$ (kcal mol$^{-1}$)", r"$\Delta E_a^\text{mech}$ (kcal mol$^{-1}$)"]
        if direction_markers is None:
            figure = plt.ScatterPlot(barr, delt,
                                     **(
                                             dict(
                                                 figure=figure,
                                                 color=color,
                                                 axes_labels=labs,
                                             ) | etc
                                     ))
        else:
            neg, pos = direction_markers
            vmin_opts = {}
            mask_negative = self.force_magnitudes < 0
            if nput.is_numeric(color[0]):
                vmin_opts['vmin'] = np.min(color)
                vmin_opts['vmax'] = np.max(color)
            mask_pos = np.where(mask_negative)
            figure = plt.ScatterPlot(barr[mask_negative], delt[mask_negative],
                                     **(
                                             dict(
                                                 figure=figure,
                                                 color=[color[i] for i in mask_pos],
                                                 axes_labels=labs,
                                                 marker=neg
                                             ) | vmin_opts | etc
                                     ))
            mask_positive = self.force_magnitudes >= 0
            mask_pos = np.where(mask_positive)
            figure = plt.ScatterPlot(barr[mask_positive], delt[mask_positive],
                                     **(
                                             dict(
                                                 figure=figure,
                                                 color=[color[i] for i in mask_pos],
                                                 axes_labels=labs,
                                                 marker=pos
                                             )  | vmin_opts | etc
                                     ))
        if plot_baseline:
            if nput.is_numeric(baseline):
                baseline = [baseline]
            if baseline_styles is None:
                baseline_styles = {'color':'gray'}
            if isinstance(baseline_styles, dict):
                baseline_styles = [baseline_styles] * len(baseline)
            for b, s in zip(baseline, baseline_styles):
                s = dict(figure=figure, linestyle='dashed') | s
                plt.Plot(figure.plot_range[0], [b, b], **s)
        return figure

    @classmethod
    def from_dataset_loader(cls,
                            loader,
                            field_map=None,
                            filter=None,
                            data_preprocessor=None,
                            fmrd_filter=None,
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
            if data_preprocessor is not None:
                data = data_preprocessor(data)
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
                    data.pop(k, None)
            if fmrd_filter is not None:
                allowed_fmrds: list[int] = fmrd_filter(data)
                if len(allowed_fmrds) == 0: continue
            else:
                allowed_fmrds = None

            if allowed_fmrds is None:
                nterms = len(fmres)
            else:
                nterms = len(allowed_fmrds)
            for k,f in field_map['vectors'].items():
                subres = data[f]
                if allowed_fmrds is not None:
                    subres = [subres[i] for i in allowed_fmrds]
                results[k].extend(subres)
            for k,f in field_map['scalars'].items():
                results[k].extend([data[f]] * nterms)

            data_ids.extend([id] * nterms)
            subres = data['force_magnitudes']
            if allowed_fmrds is not None:
                subres =  [subres[i] for i in allowed_fmrds]
            group_values, group_indices = nput.group_by(np.arange(nterms), subres)[0]
            mag_ids = np.zeros(nterms, dtype=int)
            for i,f in enumerate(group_indices): mag_ids[f] = i
            magnitude_ids.extend(mag_ids)

            # directions swap whenever groups cycle
            d_ids = np.zeros(nterms, dtype=int)
            keys, splits = nput.group_by(np.arange(nterms), mag_ids)[0]
            if len(splits[0]) > 1:
                old = splits[0][1]
                i = 0
                for i,s in enumerate(splits[0][2:]):
                    d_ids[old:s] = i + 1
                    old = s
                d_ids[old:] = i + 1
            direction_ids.extend(d_ids)

            if allowed_fmrds is None:
                filt_ids = np.arange(nterms)
            else:
                filt_ids = allowed_fmrds
            fmrd_ids.extend(filt_ids)


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
                        data.pop(k, None)
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
    def from_tree(cls, tree, depth=None, target_key='smiles', **opts):
        def loader(tree, depth, prefix=None):
            for k, v in tree.items():
                if hasattr(v, '_asdict'):
                    v = v._asdict()
                if depth is None:
                    if not isinstance(v, dict):
                        return # break entire loop
                    elif target_key in v:
                        if prefix is None:
                            yield (k,), v
                        else:
                            yield prefix+(k,), v
                    else:
                        if prefix is None:
                            yield from loader(v, None, (k,))
                        else:
                            yield from loader(v, None, prefix=prefix + (k,))
                else:
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
    def save_meta(self, file, mode=None, **etc):
        ds = self.get_data_fields()
        return utils.write_tree(file, ds, mode=mode, **etc)
    def load_meta(self, file, mode=None, **etc):
        return self.add_aggregation_fields(
            **utils.read_tree(file, mode=mode, **etc)
        )
    def save_dataset(self, file, mode=None, **etc):
        ds = self.get_reduced_dataset()
        return pipeline.write_compressed_pipeline_data(file, ds, mode=mode, **etc)

# analyses
def parse_smiles(smiles, reorder_from_atom_map=True):
    rdkit_mol = RDMolecule.parse_smiles(smiles)
    if reorder_from_atom_map:
        base_map = [a.GetAtomMapNum() for a in rdkit_mol.GetAtoms()]
        base_map = [len(base_map) + 1 if a == 0 else a for a in base_map]
        # need to use a stable sort
        rdkit_mol = Chem.RenumberAtoms(rdkit_mol, np.argsort(base_map, kind='merge').tolist())
    return rdkit_mol


def fragment_mol(mol, inds):
    submol = Chem.EditableMol(Chem.Mol())
    submol.BeginBatchEdit()
    atom_list = list(mol.GetAtoms())
    for i in inds:
        submol.AddAtom(atom_list[i])
    for i, j in itertools.combinations(range(len(inds)), 2):
        b = mol.GetBondBetweenAtoms(inds[i], inds[j])
        if b is not None:
            submol.AddBond(i, j, b.GetBondType())
    submol.CommitBatchEdit()
    return submol.GetMol()


def break_bonds(mol, bonds):
    if len(bonds) == 0:
        return {tuple(i for i, a in enumerate(mol.GetAtoms())): mol}

    bond_indices = []
    no_map = {a.GetAtomMapNum(): i for i, a in enumerate(mol.GetAtoms())}
    no_map.pop(0, None)
    for i, j in bonds:
        i, j = no_map[i + 1], no_map[j + 1]
        bond_indices.append(mol.GetBondBetweenAtoms(i, j).GetIdx())
    broke_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
    for a in broke_mol.GetAtoms(): a.SetAtomMapNum(0)
    frags = Chem.GetMolFrags(broke_mol)
    new_mols = {}
    inv_map = {i: n for n, i in no_map.items()}
    for f in frags:
        submol = fragment_mol(broke_mol, f)
        for a, i in zip(submol.GetAtoms(), f):
            a.SetAtomMapNum(inv_map.get(i, 0))
        new_mols[f] = submol
    return new_mols


def resolve_bond_type(t):
    if abs(t - 1.5) < 1e-2:
        t = Chem.BondType.names["AROMATIC"]
    elif abs(t - 2.5) < 1e-2:
        t = Chem.BondType.names["TWOANDAHALF"]
    elif abs(t - 3.5) < 1e-2:
        t = Chem.BondType.names["TWOANDAHALF"]
    else:
        t = Chem.BondType.values[int(t)]
    return t


def adjust_bond_types(mol, bond_modifications):
    rw_mol = Chem.RWMol(mol)
    no_map = {a.GetAtomMapNum(): i for i, a in enumerate(mol.GetAtoms())}
    no_map.pop(0, None)
    for (i, j), t in bond_modifications.items():
        bond = rw_mol.GetBondBetweenAtoms(no_map[i + 1], no_map[j + 1])
        bond.SetBondType(resolve_bond_type(t))
    return rw_mol.GetMol()


def reset_implicit_hydrogens(rdkit_mol):
    for atom in rdkit_mol.GetAtoms():
        if atom.GetAtomMapNum() != 0:  # only fix mapped atoms
            atom.SetNoImplicit(False)  # allow implicit Hs again
            atom.SetNumExplicitHs(0)  # clear any explicit H count
    return rdkit_mol


def split_dieneophile(smiles):
    broke = break_bonds(parse_smiles(smiles), [(0, 2), (1, 3)])
    diene, dieneophile = list(broke.values())
    dieneophile = reset_implicit_hydrogens(adjust_bond_types(dieneophile, {(2, 3): 2}))
    diene = reset_implicit_hydrogens(adjust_bond_types(diene, {(0, 4): 2, (4, 5): 1, (1, 5): 2}))
    return diene, dieneophile


def split_functional_groups(mol, preserved_sites):
    no_map = {a.GetAtomMapNum(): a for a in mol.GetAtoms()}
    no_map.pop(0, None)

    bond_indices = []
    for site, atom in no_map.items():
        if site - 1 in preserved_sites: continue
        for b in atom.GetNeighbors():
            if b.GetAtomMapNum() - 1 in preserved_sites:
                bond_indices.append([site - 1, b.GetAtomMapNum() - 1])

    pres_idx = tuple(sorted([no_map[p + 1].GetIdx() for p in preserved_sites]))

    frags = break_bonds(mol, bond_indices)
    root_frag = frags.pop(pres_idx)

    return root_frag, frags

def get_diene_atoms(diene, start, end):
    # check for the paths connecting these two atoms
    no_map = {a.GetAtomMapNum(): a for a in diene.GetAtoms()}
    vals = list(no_map.items())
    no_map.pop(0, None)

    start = no_map[start + 1]
    end = no_map[end + 1].GetIdx()

    ats = []
    for l in [4, 3]:
        paths = Chem.FindAllPathsOfLengthN(diene, l, useBonds=False, rootedAtAtom=start.GetIdx())
        for p in paths:
            p = list(p)
            if p[-1] == end:
                ats.extend(i for i in p if i not in ats)

    return [diene.GetAtomWithIdx(i).GetAtomMapNum() - 1 for i in ats]

def get_functionalization(smiles):
    diene, dieneophile = split_dieneophile(smiles)
    dats = get_diene_atoms(diene, 0, 1)
    diene_core, diene_funcs = split_functional_groups(diene, dats)
    dioph_core, dioph_funcs = split_functional_groups(dieneophile, [2, 3])

    return (diene_core, dioph_core), (diene_funcs, dioph_funcs)

def get_canonical_smiles(mol):
    mol = Chem.Mol(mol)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, 1)

fd_cache = {}
def functionalization_data(smiles):
    if smiles not in fd_cache:
        (diene_core, dioph_core), (diene_funcs, dioph_funcs) = get_functionalization(smiles)
        fd_cache[smiles] = (
            get_canonical_smiles(diene_core),
            get_canonical_smiles(dioph_core),
            tuple(
                get_canonical_smiles(v) for v in diene_funcs.values()
            ),
            tuple(
                get_canonical_smiles(v) for v in dioph_funcs.values()
            )
        )
    return fd_cache[smiles]
def functionalization_keys(smiles):
    diene, dioph, func1, func2 = functionalization_data(smiles)
    return {
        'diene': diene,
        'dienophile': dioph,
        'substitution_count': len(func1) + len(func2),
        'diene_functionalizations': func1,
        'dienophile_functionalizations': func2,
        'functional_group_types': itut.counts(func1 + func2)
    }

def fmrd_barrier(fmrd, return_bits=False):
    if hasattr(fmrd, 'fmrds'):
        fmrd = fmrd.fmrds[0]
    bits = np.array([
        fmrd.force_modified_transition_state_energy - fmrd.transition_state_energy,
        fmrd.force_modified_reactant_energy - fmrd.reactant_energy
    ]) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
    if return_bits:
        return bits
    else:
        return bits[0] - bits[1]
def pre_barrier(fmrd, return_bits=False):
    if hasattr(fmrd, 'fmrds'):
        fmrd = fmrd.fmrds[0]
    return (
        fmrd.transition_state_energy - fmrd.reactant_energy
    ) * UnitsData.convert("Hartrees", "Kilocalories/Mole")

def prep_ds(ddd,
            annotation_generator=None,
            **etc):
    if annotation_generator is None:
        annotation_generator = lambda id, data: {'smiles': data['smiles']} | functionalization_keys(data['smiles'])
    return BarrierHeightDataset.from_tree(
        ddd,
        annotation_generator=annotation_generator,
        **etc
    )
def uncompress_dataset(file, target=None):
    base, ext = os.path.splitext(file)
    if target is None:
        target = base + "_expanded.npz"
    if os.path.exists(target): return target
    ds = np.load(file)
    arrays = {k: ds[k] for k in ds.files}
    np.savez(target, **arrays)
    return target
filter_cache = {}
def load_prep_filter(file, use_cache=True,
                     update_cache=False,
                     filter=None,
                     data_preprocessor=None,
                     annotation_generator=None,
                     fmrd_filter=None,
                     **reader_opts):
    if not os.path.exists(file):
        base, ext = os.path.splitext(file)
        if len(ext) == 0:
            if os.path.exists(file + "_expanded.npz"):
                file = file + "_expanded.npz"
            else:
                file = file + ".npz"
    if not use_cache or file not in filter_cache:
        ddd_reg = pipeline.read_compressed_pipeline_data(file, **reader_opts)
        ds_reg = prep_ds(ddd_reg, filter=filter,
                         data_preprocessor=data_preprocessor,
                         fmrd_filter=fmrd_filter,
                         annotation_generator=annotation_generator)
        filter_cache[file] = ds_reg
    else:
        ds_reg = filter_cache[file]
    d2_reg = filter_d2(ds_reg)
    return ds_reg, d2_reg

def filter_d2(ds, barrier_range=[5, 80], max_delta=50, ts_thresh=0, ts_max=25, rx_thresh=-.1, rx_max=25):
    return ds.filter_by_props([
        lambda d: d['delta_transition_state'] > ts_thresh,
        lambda d: d['delta_transition_state'] < ts_max,
        lambda d: d['delta_reactant'] > rx_thresh,
        lambda d: d['delta_reactant'] < rx_max,
        lambda d: d['barrier'] > barrier_range[0],
        lambda d: d['barrier'] < barrier_range[1],
        lambda d: np.abs(d['delta']) < max_delta
    ])

def animate_rx(opt, **etc):
    return opt.animate_reactant_distortion(0, **(dict(embedding_indices=[0, 1, 2, 3, 4]) | etc))
def animate_ts(opt, **etc):
    return opt.animate_ts_distortion(0, **(dict(embedding_indices=[0, 1, 2, 3, 4]) | etc))
def animate_fmrd(opt, **etc):
    return opt.animate_fmrd_direction(0, **(dict(embedding_indices=[0, 1, 2, 3, 4]) | etc))


eh2kcal = UnitsData.convert("Hartrees", "Kilocalories/Mole")
def plot_dataset_histogram(ds, magnitudes=(50, 100, 200), use_abs=True,
                           color_generator=None,
                           bins=100,
                           figure=None,
                           property='deltas',
                           conv=None,
                           rx_max=100, ts_max=100,
                           barrier_range=[5, 80],
                           max_delta=50,
                           ts_thresh=0,
                           rx_thresh=-0.1,
                           **etc):
    if color_generator is None:
        color_generator = lambda i: plt.prep_color(palette='default', index=i)+"aa"
    figure = figure
    base_styles = (dict(
                # plot_label='rigid',
                plot_legend=True,
                legend_style={'frameon':False},
                image_size=800,
                axes_labels=[r'$\Delta\Delta E_\text{a}$', 'Counts']
            ) | etc) if figure is None else etc
    if conv is None:
        conv = UnitsData.convert("Hartrees", "Kilocalories/Mole")
    for i,m in enumerate(magnitudes):
        subds = filter_d2(ds,
                         rx_max=rx_max, ts_max=ts_max,
                           barrier_range=barrier_range,
                           max_delta=max_delta,
                           ts_thresh=ts_thresh,
                           rx_thresh=rx_thresh,
                         )[
                lambda d: (np.abs(d['force_magnitudes']) if use_abs else d['force_magnitudes']) < m+1,
                lambda d: (np.abs(d['force_magnitudes']) if use_abs else d['force_magnitudes']) > m-1
            ]
        figure = plt.HistogramPlot(
            getattr(subds, property) * conv,
            bins=bins,
            figure=figure,
            color=color_generator(i),
            label=f"{m} pN",
            **(base_styles if i == 0 else {})
        )
    return figure


def plot_dataset_stereocomp(subd1, subd2, label1='', label2=''):
    min_e1, min_e2 = np.min(subd1.reactant_energies), np.min(subd2.reactant_energies)
    min_g1 = np.where(np.abs(subd1.reactant_energies - min_e1) < 1e-8)[0]
    min_g2 = np.where(np.abs(subd2.reactant_energies - min_e2) < 1e-8)[0]
    am1, am2 = np.argmin(subd1.barriers), np.argmin(subd2.barriers)
    af1, af2 = np.argmin(subd1.reactant_energies + subd1.delta_r), np.argmin(subd2.reactant_energies + subd2.delta_r)

    mb1, mb2 = np.min(subd1.barriers), np.min(subd2.barriers)
    mbf1, mbf2 = np.min(subd1.barriers + subd1.deltas), np.min(subd2.barriers + subd2.deltas)

    meb1, meb2 = subd1.barriers[min_g1[0]], subd2.barriers[min_g2[0]]
    mef1, mef2 = subd1.barriers[af1] + subd1.deltas[af1], subd2.barriers[af2] + subd2.deltas[af2]

    mebf1, mebf2 = (
        np.min(subd1.barriers[min_g1,] + subd1.deltas[min_g1,]),
        np.min(subd2.barriers[min_g2,] + subd2.deltas[min_g2,])
    )

    dat = [
        ["Min Barrier Solvo:", mb1, mb2, mb1 - mb2],
        ["Min Barrier Force:", mbf1, mbf2, mbf1 - mbf2],
        # [subd1.barriers[am1] + subd1.deltas[am1], subd2.barriers[am2] + subd2.deltas[am2]],
        ["Min Energy Solvo:", meb1, meb2, meb1 - meb2],
        ["Min Energy Force:", mef1, mef2, mef1 - mef2],
        ["Min Energy/Barrier:", mebf1, mebf2, mebf1 - mebf2]
    ]
    dat = [
        [dd * UnitsData.convert("Hartrees", "Kilocalories/Mole") if nput.is_numeric(dd) else dd for dd in ddd]
        for ddd in dat
    ]
    return mfmt.TableFormatter(".1f",
                               headers=[
                                   [label1, 'Endo', 'Exo', 'Delta'],
                                   [label2, "kcal/mol", "kcal/mol", "kcal/mol"]
                               ],
                               column_alignments=['>', '^', '^']
                               ).format(dat)

def extract_group_means(dp_groups2, filters=None):
    dist = []
    labs = []
    if filters is None:
        filters = [
                lambda d:d['delta'] > -10,
                lambda d:np.abs(d['force_magnitudes']) > 199
            ]
    for key,ds in dp_groups2.items():
        ds = ds.__getitem__(filters)
        if len(ds) > 0:
            dist.append([
                np.mean(ds.deltas * eh2kcal),
                np.std(ds.deltas * eh2kcal)
            ])
            labs.append(key)
    return labs, dist

def format_group_table(labs, dists):
    return mfmt.TableFormatter('.2f', headers=['Diene Funcs', 'Diop Funcs.', 'Mean', 'Std']).format([
        list(k) + d
        for k, d in zip(labs, dists)
    ])
def plot_group_means(labs, dist):
    return plt.ScatterPlot(*np.array(dist).T, axes_labels=[r'$\Delta\Delta E_a$ (kcal mol$^{-1}$)', ' Standard Deviation'])
def add_keys(d2_reg):
    d2_reg_keyed = d2_reg.add_aggregation_fields(
        diop_func=np.array([" ".join(d) for d in d2_reg.dienophile_functionalizations]),
        dien_func=np.array([" ".join(d) for d in d2_reg.diene_functionalizations]),
        d_path=np.array(["/".join(d) for d in d2_reg.data_ids])
        # pool=None
    )
    return d2_reg_keyed

def get_dienophile_groups(d2_reg):
    d2_reg_keyed = add_keys(d2_reg)
    dp_groups = {
        k: d.group_by_props('smiles')
        for k, d in d2_reg_keyed.group_by_props('diop_func').items()
    }
    dp_groups2 = d2_reg_keyed.group_by_props(['dien_func', 'diop_func'])
    return d2_reg_keyed, dp_groups, dp_groups2


import scipy


def kde_plot(data, filled=True, figure=None, color=None, plot_range=None, scaling=None, **opts):
    kde = scipy.stats.gaussian_kde(data, bw_method=.1)
    if plot_range is None:
        if figure is None:
            dm, DM = data.min(), data.max()
            r = DM - dm
            dm = dm - r * .2
            DM = DM + r * .2
        else:
            plot_range = figure.plot_range
            dm, DM = plot_range[0]
    else:
        dm, DM = plot_range[0]
    x_grid = np.linspace(dm, DM, 500)
    if scaling is None:
        scaling = len(data)
    y_density = kde(x_grid) * scaling
    if filled:
        if color is not None:
            acolor = plt.prep_color(color, alpha=.1)
        else:
            acolor = plt.prep_color(palette='default', index=0, alpha=.1)
        figure = plt.FilledPlot(x_grid, y_density, color=acolor, figure=figure,
                                plot_range=plot_range,
                                **(opts | dict(label=None)))
    return plt.Plot(x_grid, y_density, figure=figure, color=color,
                    plot_range=plot_range,
                    **opts)

def get_freqs(mol):
    return mol.get_normal_modes().freqs * UnitsData.hartrees_to_wavenumbers
def get_compliances(mol):
    mol_int = mol.modify(
        internals={'primitives': [tuple(x) for x in nput.combination_indices(len(mol.atoms), 2)]}
    )
    mol_int.potential_derivatives = [0, mol.potential_derivatives[1]]
    H = mol_int.get_internal_potential_derivatives(order=2)[1]
    return np.diag(nput.frac_powh(H, -1))
_DEBUG_PRINT_KEYS = True
def freq_data(mol, freq_filter=None, freq_gen=None, fcache=None):
    if hasattr(mol, 'get_normal_modes'):
        mol = [mol]
    if fcache is None:
        fcache = {}
    freq_lists = []
    for k, m in enumerate(mol):
        ix = m.id if hasattr(m, 'id') else k
        if ix not in fcache:
            if _DEBUG_PRINT_KEYS:
                print(ix)
            if freq_gen is None:
                freq_gen = get_freqs
            fcache[ix] = freq_gen(m)
        freq_lists.append(fcache[ix])
    freqs = np.concatenate(freq_lists)
    if freq_filter is not None:
        freqs = freqs[freq_filter(freqs)]
    return freqs
def freq_plot(mol, freq_filter=None, freq_gen=None, **opts):
    return kde_plot(freq_data(mol, freq_filter, freq_gen), **opts)
def freq_comp_data(opt, freq_filter=None, freq_gen=None, fcache=None):
    if hasattr(opt, 'deltas'):
        ds = opt
        opt = list(iter(opt))
        for o, i in zip(opt, ds.data_ids):
            o.data_id = i
    if (
            hasattr(opt, 'optimizer')
            or hasattr(opt, 'rs')
    ): opt = [opt]
    if fcache is None:
        fcache = {}
    ids = [o.data_id if hasattr(o, 'data_id') else None for o in opt]
    opt = [o.optimizer if hasattr(o, 'optimizer') else o for o in opt]
    rs_mols = [o.rs for o in opt]
    for m, i in zip(rs_mols, ids): m.id = (i, "rs")
    ts_mols = [o.ts for o in opt]
    for m, i in zip(ts_mols, ids): m.id = (i, "ts")
    rs_data = freq_data(rs_mols,

                        freq_filter=freq_filter,
                        freq_gen=freq_gen,
                        fcache=fcache
                        )
    ts_data = freq_data(ts_mols,
                        freq_filter=freq_filter,
                        freq_gen=freq_gen,
                        fcache=fcache
                        )
    return rs_data, ts_data


def freq_comp_plot(opt, freq_filter=None, figure=None, **opts):
    if hasattr(opt, 'deltas'):
        ds = opt
        opt = list(iter(opt))
        for o, i in zip(opt, ds.data_ids):
            o.data_id = i
    if (
            hasattr(opt, 'optimizer')
            or hasattr(opt, 'rs')
    ): opt = [opt]
    ids = [o.data_id if hasattr(o, 'data_id') else None for o in opt]
    opt = [o.optimizer if hasattr(o, 'optimizer') else o for o in opt]
    rs_mols = [o.rs for o in opt]
    for m, i in zip(rs_mols, ids): m.id = (i, "rs")
    figure = freq_plot(rs_mols,
                       **dict(
                           freq_filter=freq_filter,
                           figure=figure,
                           label='reactant',
                           plot_legend=True,
                           legend_style={'frameon': False},
                           color=plt.prep_color(palette='default', index=0)
                       ) | opts
                       )
    ts_mols = [o.ts for o in opt]
    for m, i in zip(ts_mols, ids): m.id = (i, "ts")
    figure = freq_plot(ts_mols,
                       **dict(
                           freq_filter=freq_filter,
                           figure=figure,
                           label='ts',
                           color=plt.prep_color(palette='default', index=2)
                       ) | opts
                       )
    return figure
def plot_dataset_freq_comp(ds, wm=None, **etc):
    opts = list(iter(ds))
    for o, i in zip(opts, ds.data_ids):
        o.data_id = i
    return freq_comp_plot(
        opts,
        **(
                dict(
                    plot_label=(
                                   f"Mean: {wm[0]:.2f} kcal/mol Std: {wm[1]:.2f} kcal/mol"
                                   if wm is not None else None
                    ),
                    display_format='svg'
                ) | etc
        )
    )

def extract_group_compliances(groups, nmax=None):
    fcache = {}
    return {
        ds.d_path[0]: freq_comp_data(ds, freq_gen=get_compliances, fcache=fcache)
        for k, ds in (
            itertools.islice(groups.items(), nmax)
                if nmax is not None else
            groups.items()
        )
    }

def save_compliance_dataset(d2_reg, file, nmax=None):
    wah = get_dienophile_groups(d2_reg)
    dp_wah = wah[0].group_by_props('d_path')
    data = extract_group_compliances(dp_wah, nmax)
    return utils.write_tree(file, data)

def _flatten_weight_block(rs_blocks):
    blocks = []
    for r, w in rs_blocks.values():
        blocks.append(
            np.stack(
                [r, np.repeat(w, len(r))],
                axis=1
            )
        )
    if len(blocks) == 0:
        return None
    else:
        return np.concatenate(blocks, axis=0)
def _get_group_blocks(ds_paths, compliance_data, idx, flatten=True):
    subblocks = {}
    for p in ds_paths:
        if 'pipeline_data' in p:
            for tag in [
                'random',
                'internals',
                "useint",
                'rigid',
            ]:
                p = p.replace(f"pipeline_data_{tag}", "pipeline_data")
        if p not in subblocks:
            if p in compliance_data:
                    subblocks[p] = [compliance_data[p][idx], 1]
        else:
            subblocks[p][1] += 1
    if flatten:
        subblocks = _flatten_weight_block(subblocks)
    return subblocks
def get_compliance_groups(d2_groups, compliance_data, flatten=True):
    keys, mean_data = extract_group_means(d2_groups)

    rs_blocks = []
    subkeys = []
    mask = np.full(len(keys), False)
    for i,k in enumerate(keys):
        block = _get_group_blocks(d2_groups[k].d_path, compliance_data, 0, flatten=flatten)
        if block is not None:
            rs_blocks.append(block)
            mask[i] = True
            subkeys.append(k)
    ts_blocks = []
    for i,k in enumerate(keys):
        block = _get_group_blocks(d2_groups[k].d_path, compliance_data, 1, flatten=flatten)
        if block is not None:
            ts_blocks.append(block)
            mask[i] = True
            subkeys.append(k)
    means = np.array([m[0] for i, m in enumerate(mean_data) if mask[i]])
    stds = np.array([m[1] for i, m in enumerate(mean_data) if mask[i]])
    return (subkeys, means, stds), rs_blocks, ts_blocks

def get_compliance_means(rs_blocks, ts_blocks, use_abs=True, thresh=2.5e4):
    if use_abs:
        rs_blocks = [np.abs(x) for x in rs_blocks]
        ts_blocks = [np.abs(x) for x in ts_blocks]
        mask_r = [r[:, 0] < thresh for r in rs_blocks]
        mask_t = [t[:, 0] < thresh for t in ts_blocks]
    else:
        mask_r = [np.abs(r[:, 0]) < thresh for r in rs_blocks]
        mask_t = [np.abs(t[:, 0]) < thresh for t in ts_blocks]
    rs_means2 = [
        np.average(r[m][:, 0], weights=r[m][:, 1])
        for r,m in zip(rs_blocks, mask_r)
    ]
    ts_means2 = [
        np.average(r[m][:, 0], weights=r[m][:, 1])
        for r, m in zip(ts_blocks, mask_t)
    ]
    return np.array(rs_means2), np.array(ts_means2)

def plot_compliance_means(
        means, rs_means2, ts_means2,
        rs_styles='auto',
        ts_styles='auto',
        figure=None,
        **styles
):
    if dev.str_is(rs_styles, 'auto'):
        rs_styles = {}
    if rs_styles is not None:
        rs_styles = rs_styles | styles
        figure = plt.ScatterPlot(
            rs_means2,
            means[1],
            figure=figure,
            **rs_styles
        )
    if dev.str_is(ts_styles, 'auto'):
        ts_styles = {'color':plt.prep_color(palette='default', alpha=.3, index=1)}
    if ts_styles is not None:
        ts_styles = ts_styles | styles
        figure = plt.ScatterPlot(
            ts_means2,
            means[1],
            figure=figure,
            **ts_styles
        )
    return figure

def get_compliance_data(d2_reg, compliance_data):
    wah = get_dienophile_groups(d2_reg)
    means, rs_groups, ts_groups = get_compliance_groups(wah[2], compliance_data)
    wop = get_compliance_means(rs_groups, ts_groups)
    return means, wop[0], wop[1]

def get_compliance_fit(compliances, barriers, include_intercept=True):
    if include_intercept:
        def model_func(slope, intercept, x):
            return slope * x + intercept
    else:
        def model_func(slope, x):
            return slope * x
    def residual_func(params, x, y):
        return model_func(*params, x) - y

    if include_intercept:
        initial_guess = np.array([-1000.0, 0])
    else:
        initial_guess = np.array([-1000.0])

    result = least_squares(residual_func, initial_guess, args=(compliances, barriers))

    optimized_params = result.x  # Array of best-fit [slope, intercept]
    residuals = residual_func(optimized_params, compliances, barriers)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((barriers - np.mean(barriers)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    model = functools.partial(model_func, *optimized_params)
    return optimized_params, model, r_squared