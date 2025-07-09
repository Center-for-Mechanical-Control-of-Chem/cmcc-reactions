
import numpy as np
import collections
import json
import pickle
import base64
import numbers

__all__ = [
    # "write_distortion_data",
    # "read_distortion_data"
]

def compress_tree(tree_obj, top_level=True):
    subtrees = {
        'key_map': {}
    }
    for k,(s,v) in enumerate(tree_obj.items()):
        subtrees['key_map'][k] = s
        if isinstance(v, dict):
            subtrees[k] = compress_tree(v, top_level=False)
        elif isinstance(v, (int, bool, float, str, numbers.Number)):
            subtrees[k] = ((0,-1), np.array([v]))
        else:
            v = np.asanyarray(v)
            if v.shape == ():
                subtrees[k] = ((0,-1), np.array([v]))
            else:
                subtrees[k] = (v.shape + (-1,), v.flatten())

    return merge_trees(subtrees, top_level=top_level)

def merge_trees(subtrees, top_level=True):
    key_lists = {
        'visited_keys': subtrees.pop('visited_keys', []),
        'key_map': subtrees.pop('key_map', {}),
        # 'key_depths':[]
    }
    key_map = key_lists['key_map']
    inv_map = {k:v for v,k in key_map.items()}

    for k,s in subtrees.items():
        key_lists['visited_keys'].append(k)
        if isinstance(s, dict):
            s_map = s.pop('key_map', {})
            for vv,sk in s_map.items():
                if sk not in inv_map:
                    n = max(key_map.keys()) + 1
                    key_map[n] = sk
                    inv_map[sk] = n
            for sk,v in s.items():
                if sk == 'visited_keys':
                    # if not bottom_level:
                    #     key_lists['visited_keys'].append(-1)

                    key_lists['visited_keys'].extend(
                        inv_map[s_map[vv]]
                            if vv >= 0 else
                        vv
                            for vv in v
                    )
                else:
                    sk = inv_map[s_map[sk]]
                    if sk not in key_lists: key_lists[sk] = []
                    key_lists[sk].append(v)
        else:
            if k not in key_lists: key_lists[k] = []
            key_lists[k].append(s)
    if not top_level:
        key_lists['visited_keys'].append(-1)

    for key,value_list in key_lists.items():
        if key in {'key_map', 'visited_keys'}:
            key_lists[key] = value_list
            continue
        shapes = []
        for v in value_list:
            shapes.extend(v[0])
        values = np.concatenate([v[1] for v in value_list])
        key_lists[key] = (shapes, values)

    return key_lists

def decompress_tree(serial_tree):
    tree = {}
    tree_stack = collections.deque()
    key_map = serial_tree.pop('key_map')
    block_pointers = {}
    for k in serial_tree['visited_keys']:
        if k >= 0:
            s = key_map[k]
            data = serial_tree.get(k)
            if data is not None:
                if k not in block_pointers:
                    block_pointers[k] = (0, 0)
                shape_pointer, array_pointer = block_pointers[k]
                shape_data, array_data = data
                shape_offset = shape_pointer
                for shape_offset in range(shape_pointer, len(shape_data)):
                    if shape_data[shape_offset] < 0: break
                shape = tuple(shape_data[shape_pointer:shape_offset])
                if shape == (0,):
                    block_size = 1
                    shape = ()
                else:
                    block_size = np.prod(shape, dtype=int)
                arr = array_data[array_pointer:array_pointer+block_size].reshape(shape)
                block_pointers[k] = (shape_offset+1, array_pointer + block_size)
                tree[s] = arr
            else:
                tree[s] = {}
                tree_stack.append(tree)
                tree = tree[s]
        else:
            tree = tree_stack.pop()
    return tree

class BaseEncoder(json.JSONEncoder):
    def __init__(self, *args, allow_pickle=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_pickle=allow_pickle
    @classmethod
    def unpickle_object(cls, b64stream:str):
        dump = base64.b64decode(b64stream.encode('ascii'))
        return pickle.loads(dump)
    @classmethod
    def check_unpickle_dict(cls, d:dict):
        if len(d) == 1 and list(d.keys()) == ['/pickled_object/']:
            return True, cls.unpickle_object(list(d.values())[0])
        else:
            return False, d

    def pickle_object(self, obj):
        dump = pickle.dumps(obj)
        dump = base64.b64encode(dump).decode('ascii')
        return {'/pickled_object/':dump}
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            if self.allow_pickle:
                try:
                    stream = json.JSONEncoder.default(self, obj)
                except TypeError:
                    stream = self.pickle_object(obj)
                return stream
            else:
                return json.JSONEncoder.default(self, obj)


def write_tree(file, data, compress=True, mode='npz', **opts):
    if mode == 'json':
        if not hasattr(file, 'write'):
            with open(file, 'w+') as fp:
                json.dump(data, fp, cls=BaseEncoder, **opts)
        else:
            json.dump(data, file, cls=BaseEncoder, **opts)
    else:
        if compress:
            compressed = compress_tree(data)
        else:
            compressed = data
        key_names = list(compressed['key_map'].values())
        index_remapping = {k:i for i,k in enumerate(compressed['key_map'].keys())}
        visited_keys = [index_remapping[i] if i >= 0 else i for i in compressed['visited_keys']]
        arrays = {}
        shapes = []
        array_keys = []
        for k in compressed['key_map'].keys():
            if k in compressed:
                shape_data, array_data = compressed[k]
                shapes.append(len(shape_data))
                shapes.extend(shape_data)
                i = index_remapping[k]
                arrays[f'arr_{i}'] = array_data
                array_keys.append(i)
        return np.savez(
            file,
            shapes=shapes,
            key_names=key_names,
            array_keys=array_keys,
            visited_keys=visited_keys,
            **arrays
        )
def normalize_tree(data):
    if isinstance(data, dict):
        was_pick, obj = BaseEncoder.check_unpickle_dict(data)
        if not was_pick:
            return {k:normalize_tree(o) for k,o in data.items()}
        else:
            return obj
    else:
        return data

def read_tree(file, decompress=True, mode='npz'):
    if mode == 'json':
        if not hasattr(file, 'write'):
            with open(file) as fp:
                data = json.load(fp)
        else:
            data = json.load(file)
        return normalize_tree(data)
    else:
        zdata = np.load(file)
        key_names = zdata['key_names']
        visited_keys = zdata['visited_keys']
        shapes = zdata['shapes']
        array_keys = zdata['array_keys']
        compressed = {
            'visited_keys':visited_keys,
            'key_map':{
                i: k for i, k in enumerate(key_names)
            }
        }

        shape_pointer = 0
        for k in array_keys:
            ls = shapes[shape_pointer]
            new_pointer = shape_pointer+1+ls
            shape = shapes[shape_pointer+1:new_pointer]
            shape_pointer = new_pointer
            array = zdata[f'arr_{k}']
            compressed[k] = (shape, array)

        if decompress:
            return decompress_tree(compressed)
        else:
            return compressed

def write_distortion_data(file, data, validate=True):
    if validate:
        validate_distortion_data(data)

    return write_tree(file, data)


schemas = {}
class DistortionDataValidationError(ValueError):
    ...

def validate_instance_type(value, type, validate_instances=False):
    if isinstance(type, str):
        if validate_instances:
            try:
                validate_base_schema(type, schemas[type], value)
            except DistortionDataValidationError:
                return False
        else:
            return type
    elif isinstance(type, list):
        try:
            viter = iter(value)
        except TypeError:
            return False
        else:
            return all(
                validate_instance_type(v, type[0], validate_instances=True)
                for v in viter
            )
    else:
        return isinstance(value, type)

def validate_base_schema(name, schema, instance):
    missing_keys = []
    for k in schema.get('required', []):
        if isinstance(k, str):
            if k not in instance:
                missing_keys.append(k)
        else:
            if all(sk not in instance for sk in k):
                missing_keys.append(' or '.join(k))
    if len(missing_keys) > 0:
        missing_keys = "\n".join(missing_keys)
        raise DistortionDataValidationError(f"instance of schema `{name}` missing keys {missing_keys}")

    bad_types = []
    extra_keys = []
    delayed_validations = []
    for k,v in instance:
        type = schema.get(k)
        if type is None:
            if not schema.get('$allow_extra_keys', False):
                extra_keys.append(k)
                continue
            else:
                type = schema.get('$default')
                if type is None: continue

        delayed_validations.append([k, v, type])


    if len(extra_keys) > 0:
        bad_types = "\n".join(f"{k}: expected {t} got {v}" for k,v,t in bad_types)
        raise DistortionDataValidationError(
            f"instance of schema `{name}` had mis-typed values {bad_types}"
        )
    if len(bad_types) > 0:
        bad_types = "\n".join(f"{k}: expected {t} got {v}" for k,v,t in bad_types)
        raise DistortionDataValidationError(
            f"instance of schema `{name}` had mis-typed values {bad_types}"
        )




def validate_structure_schema(instance):
    coords = instance['coordinates']
    hess = instance.get('hessian')
    if hess is not None:
        hess = np.asarray(hess)
        if hess.shape[0] != hess.shape[1]:
            raise DistortionDataValidationError("Hessian not square")
        ncrds = np.asanyarray(coords).flatten().shape[0]
        if ncrds != hess.shape[0]:
            raise DistortionDataValidationError("Hessian shape does not match coords")


schemas['structure'] = {
    'coordinates':np.ndarray,
    'energy':float,
    'hessian':np.ndarray,
    "$validator": validate_structure_schema,
    "$required": ['coordinates']
}

schemas['results'] = {
    'reactant':'$structure',
    'transition_state':'$structure',
}

class distortion_types:
    RigidSinglePoints = 'single_points'
    RelaxedSinglePoints = 'relaxed_single_points'
    MEPSampling = 'mep'

schemas['distortion'] = {
    'results':['$structure'],
    'spec':[int],
    'distortion':np.ndarray,
    'magnitudes':np.ndarray,
    'gamma':float,
    'method':distortion_types,
    # '$validator':validate_distortion,
    '$required':['results']
}
def validate_solvothermal_distortion(instance):
    ...
def validate_internal_distortion(instance):
    ...
def validate_direction_distortion(instance):
    ...

def validate_lot(instance):
    ...

schemas['lot'] = {
    'atoms':np.ndarray,
    '$default':'$results',
    '$allow_extra_keys':True,
    "$validator": validate_lot,
    "$required": ['coordinates']
}