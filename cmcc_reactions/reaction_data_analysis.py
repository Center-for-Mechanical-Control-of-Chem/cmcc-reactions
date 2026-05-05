
import collections
import numpy as np

from . import reaction_data_schema as schema
from . import utils
from . import generate_reaction_products as gen_prods

from McUtils.Data import UnitsData
import McUtils.Numputils as nput
import McUtils.Plots as plt
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

def get_critical_points(trajectory, energies=None, initial=None):
    if energies is None:
        energies = [g.calculate_energy() for g in trajectory]

    product_idx = np.argmin(energies)
    ts_idx = np.argmax(energies)
    if product_idx > ts_idx:
        react_idx = np.argmin(energies[:ts_idx])
    else:
        react_idx = np.argmin(energies[ts_idx:])
    return energies, (ts_idx, react_idx, product_idx)

class DielsAlderReactionTrajectory:
    def __init__(self, atoms, structures,
                 energies=None,
                 ts_index=None,
                 reactant_index=None,
                 product_index=None,
                 energy_evaluator=None,
                 distance_units=None,
                 diene_atoms=(2, 3, 4, 5),
                 dienophile_atoms=(0, 1)
                 ):
        self.atoms = atoms
        self.structures = np.asanyarray(structures)
        self._energies = energies
        self._mols = [None] * len(self.atoms)
        self._ts_idx = ts_index
        self._reactant_idx = reactant_index
        self._product_idx = product_index
        self.energy_evaluator = energy_evaluator
        self.distance_units = distance_units
        self.diene_atoms = diene_atoms
        self.dienophile_atoms = dienophile_atoms

    def load_mol(self, i):
        struct = self.structures[i]
        if self.distance_units is not None:
            struct = struct * UnitsData.convert(self.distance_units, "BohrRadius")
        return Molecule(
            self.atoms,
            struct,
            energy_evaluator=self.energy_evaluator
        )

    @property
    def energies(self):
        if self._energies is None:
            self._energies = [g.calculate_energy() for g in self.mols]
        return self._energies

    @property
    def mols(self):
        if self._mols is None:
            self._mols = [self.load_mol(i) for i in range(len(self.structures))]
        return self._mols

    @classmethod
    def from_trajectory_data(cls,
                             trajectory_data,
                             structures=None,
                             energies=None,
                             energy_evaluator=None,
                             which='final',
                             **etc):
        if isinstance(trajectory_data, str):
            trajectory_data = utils.read_namedtuple(trajectory_data, gen_prods.ReoptimizedTrajectoryData)

        if energies is None:
            if which == 'final':
                energies = trajectory_data.final_energies
            else:
                energies = trajectory_data.initial_energies

        if structures is None:
            if which == 'final':
                structures = trajectory_data.final_trajectory
            else:
                structures = trajectory_data.initial_trajectory

        if energy_evaluator is None:
            energy_evaluator = trajectory_data.evaluator
        return cls(
            trajectory_data.atoms,
            structures=structures,
            energies=energies,
            energy_evaluator=energy_evaluator,
            **etc
        )

def centroid_distance(traj, bonds):
    traj = np.asanyarray(traj)
    a1, a2 = np.array(bonds).T
    return nput.pts_norms(
        np.average(traj[:, a1], axis=-1),
        np.average(traj[:, a2], axis=-1)
    )

def dienophile_distance(traj, bonds):
    traj = np.asanyarray(traj)
    _, (i, j) = np.array(bonds).T
    return nput.pts_norms(traj[:, i], traj[:, j])

def incremental_rmsds(traj, bonds=None, sel=None):
    traj = np.asanyarray(traj)
    if sel is None and bonds is not None:
        sel = np.asanyarray(bonds).flatten()
    if sel is not None:
        traj = traj[..., sel, :]
    rmsds = np.diff(traj, axis=0).reshape((len(traj)-1, -1))
    return np.cumsum(np.concatenate([[0], rmsds]), axis=0)

metric_label_map = {
    centroid_distance: 'Centroid Distance',
    dienophile_distance: r'$r_{\text{C=C}}$'
}
def plot_trajectory(
        coordinates, energies,
        use_rmsd=None,
        distance_metric=None,
        metric_label=None,
        **opts):
    min_e = np.min(np.concatenate([traj_data.final_energies[:5], traj_data.initial_energies[:5]]))
    if distance_metric is None:
        distance_metric = dienophile_distance
    elif use_rmsd is None:
        use_rmsd = False
    if use_rmsd:
        if metric_label is None:
            metric_label = "RMSD"
        x1 = np.array(traj_data.final_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
    else:
        if metric_label is None:
            metric_label = metric_label_map.get(distance_metric)
            if metric_label is None:
                metric_label = distance_metric.__name__
        x1 = distance_metric(traj_data.final_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
                                                                                               "Angstroms")
    return plt.Plot(
        x1,
        (traj_data.final_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
        **(dict(
            axes_labels=[
                metric_label + r" ($\AA$)",
                "E (kcal mol$^{-1}$)"
            ]
        ) | opts)
    )

def plot_comp_traj(traj_data, comp_data=None,
                   use_rmsd=None,
                   distance_metric=None,
                   metric_label=None,
                   **opts):
    min_e = np.min(np.concatenate([traj_data.final_energies[:5], traj_data.initial_energies[:5]]))
    if distance_metric is None:
        distance_metric = dienophile_distance
    elif use_rmsd is None:
        use_rmsd = False
    if use_rmsd:
        if metric_label is None:
            metric_label = "RMSD"
        x1 = np.array(traj_data.final_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
    else:
        if metric_label is None:
            metric_label = metric_label_map.get(distance_metric)
            if metric_label is None:
                metric_label = distance_metric.__name__
        x1 = distance_metric(traj_data.final_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
                                                                                               "Angstroms")
    f1 = plt.Plot(
        x1,
        (traj_data.final_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
        **(dict(
            axes_labels=[
                metric_label + r" ($\AA$)",
                "E (kcal mol$^{-1}$)"
            ]
        ) | opts)
    )
    if comp_data is None:
        if use_rmsd:
            x2 = np.array(traj_data.initial_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
        else:
            x2 = distance_metric(traj_data.initial_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
                                                                                                     "Angstroms")
        plt.Plot(
            x2,
            (traj_data.initial_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
            figure=f1,
            linestyle='dashed'
        )
    else:
        if use_rmsd:
            x2 = np.array(comp_data.final_rmsds) * UnitsData.convert("BohrRadius", "Angstroms")
        else:
            x2 = distance_metric(comp_data.final_trajectory, [[0, 2], [1, 3]]) * UnitsData.convert("BohrRadius",
                                                                                                   "Angstroms")
        plt.Plot(
            x2,
            (comp_data.final_energies - min_e) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
            figure=f1,
            linestyle='dashed'
        )
    return f1
