
import numpy as np
import collections
import json
import pickle
import base64
import numbers
import glob
import os
import io

import McUtils.Devutils as dev

__all__ = [
    "read_tree",
    "write_tree",
    "write_json",
    "read_json",
    "write_namedtuple",
    "dumps_namedtuple",
    "make_namedtuple",
    "read_namedtuple",
    "loads_namedtuple",
    "isnamedtupleinstance"
]

def dictify_lists(tree:dict):
    tree = tree.copy()
    for k,subtree in tree.items():
        if isinstance(subtree, dict):
            tree[k] = dictify_lists(subtree)
        elif isinstance(subtree, (list, tuple)):
            if all(isinstance(d, dict) for d in subtree):
                tree[k] = {
                    f'_list_item_{i}':dictify_lists(v)
                    for i,v in enumerate(subtree)
                }
                tree[k]['_num_list_items'] = len(subtree)
            elif (
                    isinstance(subtree, (list, tuple))
                    and dev.is_list_like(subtree[0])
                    and len(np.unique([len(y) for y in subtree])) > 1
            ):
                tree[k] = {
                    f'_list_item_{i}': v
                    for i, v in enumerate(subtree)
                }
                tree[k]['_num_list_items'] = len(subtree)
    return tree
def compress_tree(tree_obj, top_level=True, prep_tree=True):
    if prep_tree:
        tree_obj = dictify_lists(tree_obj)

    subtrees = {
        'key_map': {}
    }
    for k,(s,v) in enumerate(tree_obj.items()):
        subtrees['key_map'][k] = s
        if isinstance(v, dict):
            subtrees[k] = compress_tree(v, top_level=False, prep_tree=False)
        elif isinstance(v, (int, bool, float, str, numbers.Number)):
            subtrees[k] = ((0,-1), np.array([v]))
        elif v is None:
            subtrees[k] = ((0,-1), np.array([np.nan]))
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

def undictify_lists(tree:dict):
    tree = tree.copy()
    for k,subtree in tree.items():
        if isinstance(subtree, dict):
            if '_num_list_items' in subtree:
                tree[k] = [
                    subtree[f'_list_item_{i}']
                    for i in range(subtree['_num_list_items'])
                ]
            else:
                tree[k] = undictify_lists(subtree)
    return tree
def decompress_tree(serial_tree, unprep_tree=True):
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
                if arr.ndim == 0:
                    if np.issubdtype(arr.dtype, np.dtype(float)) and np.isnan(arr):
                        arr = None
                    elif np.issubdtype(arr.dtype, np.dtype(str)):
                        arr = arr.tolist()
                tree[s] = arr
            else:
                tree[s] = {}
                tree_stack.append(tree)
                tree = tree[s]
        else:
            tree = tree_stack.pop()
    if unprep_tree:
        tree = undictify_lists(tree)
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

def write_json(file, data, **opts):
    if not hasattr(file, 'write'):
        with open(file, 'w+') as fp:
            json.dump(data, fp, cls=BaseEncoder, **opts)
    else:
        json.dump(data, file, cls=BaseEncoder, **opts)
    return file

def read_json(file, normalize=True, **opts):
    if not hasattr(file, 'read'):
        with open(file) as fp:
            data = json.load(fp)
    else:
        data = json.load(file)
    if normalize:
        return normalize_tree(data)
    else:
        return data

def write_tree(file, data, compress=None, mode=None, precompression_function=None, **opts):
    if mode is None:
        if isinstance(file, str) and os.path.splitext(file)[-1] == '.json':
            mode = 'json'
        else:
            mode = 'npz'
    if mode == 'json':
        if not hasattr(file, 'write'):
            with open(file, 'w+') as fp:
                json.dump(data, fp, cls=BaseEncoder, **opts)
        else:
            json.dump(data, file, cls=BaseEncoder, **opts)
    else:
        if compress is None:
            compress = True
        if compress:
            if precompression_function is not None:
                data = precompression_function(data)
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
def dumps_tree(data, compress=None, mode='json', **opts):
    if compress is None:
        compress = mode != 'json'
    buf = io.StringIO() if mode == 'json' else io.BytesIO()
    write_tree(buf, data, compress=compress, mode=mode, **opts)
    buf.seek(0)
    return buf.read()
def normalize_tree(data):
    if isinstance(data, dict):
        was_pick, obj = BaseEncoder.check_unpickle_dict(data)
        if not was_pick:
            return {k:normalize_tree(o) for k,o in data.items()}
        else:
            return obj
    else:
        return data

def read_tree(file, decompress=None, mode=None, decompression_function=None, **opts):
    if mode is None:
        if isinstance(file, str) and os.path.splitext(file)[-1] == '.json':
            mode = 'json'
        else:
            mode = 'npz'
    if mode == 'json':
        if not hasattr(file, 'read'):
            with open(file) as fp:
                data = json.load(fp, **opts)
        else:
            data = json.load(file, **opts)
        return normalize_tree(data)
    else:
        if decompress is None: decompress = True
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
            data = decompress_tree(compressed)
            if decompression_function is not None:
                data = decompression_function(data)
            return data
        else:
            return compressed
def loads_tree(data, decompress=None, mode='npz', **opts):
    buf = io.StringIO() if isinstance(data, str) else io.BytesIO()
    buf.write(data)
    buf.seek(0)
    return read_tree(buf, decompress=decompress, mode=mode, **opts)

namedtuple_registry = {}
namedtuple_defaults = {}
def register_namedtuple(type, defaults=None):
    namedtuple_registry[type.__name__] = type
    if defaults is not None:
        namedtuple_defaults[type] = defaults
    return type
def prep_compressed_namedtuple_data(data):
    for k, v in data.items():
        if k.endswith('_settings') and isinstance(v, dict):
            data[k] = {k + tag:d for tag, d in v.items()}
    return data
def write_namedtuple(file, obj, compress=None, mode=None, **opts):
    d = obj._asdict() | {"_type":type(obj).__name__}
    return write_tree(file, d, compress=compress, mode=mode, precompression_function=prep_compressed_namedtuple_data, **opts)
def dumps_namedtuple(obj, compress=None, mode='json', **opts):
    d = obj._asdict() | {"_type":type(obj).__name__}
    return dumps_tree(d, compress=compress, mode=mode, precompression_function=prep_compressed_namedtuple_data, **opts)
def make_namedtuple(obj, nt_type=None, key=None, in_place=False):
    if key is not None:
        if not isinstance(key, str):
            key = [key]
        for k in key:
            obj = obj[k]
    if not in_place: obj = obj.copy()
    tn = obj.pop('_type', None)
    if nt_type is None:
        if tn is None: raise ValueError("can't load `namedtuple` without type name")
        nt_type = tn
    if isinstance(nt_type, str):
        nt_type = namedtuple_registry[nt_type]
    defaults = namedtuple_defaults.get(nt_type)
    if defaults is not None:
        obj = defaults | obj

    return nt_type(**obj)
def decompress_namedtuple_data(data):
    for k, v in data.items():
        if k.endswith('_settings') and isinstance(v, dict):
            tl = len(k)
            data[k] = {
                tag[tl:] if tag.startswith(k) else tag:d
                for tag, d in v.items()
            }
    return data
def read_namedtuple(file, nt_type=None, decompress=None, mode=None, key=None, **opts):
    obj = read_tree(file, decompress=decompress, mode=mode, decompression_function=decompress_namedtuple_data, **opts)
    return make_namedtuple(obj, nt_type=nt_type, key=key, in_place=True)
def loads_namedtuple(buf, nt_type=None, decompress=None, mode='json', key=None, **opts):
    obj = loads_tree(buf, decompress=decompress, decompression_function=decompress_namedtuple_data, mode=mode, **opts)
    return make_namedtuple(obj, nt_type=nt_type, key=key, in_place=True)
def isnamedtupleinstance(obj, nt_types):
    if not isinstance(nt_types, tuple):
        nt_types = (nt_types,)
    return (
            isinstance(obj, nt_types)
            or any(
                all(
                    hasattr(obj, k)
                    for k in nt_type._fields
                )
                for nt_type in nt_types
            )
    )

def construct_json_file_tree(top_dir, js_patterns="**/*.json", recursive=True):
    tree = {}
    if isinstance(js_patterns, str):
        js_patterns = [js_patterns]
    files = []
    for pattern in js_patterns:
        files.extend(glob.glob(pattern, root_dir=top_dir, recursive=recursive))
    for f in files:
        segments = dev.split_path(f)
        subtree = tree
        for s in segments[:-1]:
            if s not in subtree:
                subtree[s] = {}
            subtree = subtree[s]
        name = os.path.splitext(segments[-1])[0]
        subtree[name] = dev.read_json(os.path.join(top_dir, f))
    return tree

def annotate_json_file_tree(top_dir, js_patterns, get_annotations=None, recursive=True, **opts):
    tree = {}
    if isinstance(js_patterns, str):
        js_patterns = [js_patterns]
    files = []
    for pattern in js_patterns:
        files.extend(glob.glob(pattern, root_dir=top_dir, recursive=recursive))
    for f in files:
        f = os.path.join(top_dir, f)
        dev_tree = dev.read_json(f)
        dev_tree.update(**opts)
        if get_annotations is not None:
            dev_tree.update(**get_annotations(f))
        dev.write_json(f, dev_tree)
    return tree