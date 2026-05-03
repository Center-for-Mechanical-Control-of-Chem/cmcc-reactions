
import numpy as np
from .utils import write_tree, read_tree

__all__ = [
    "validate_distortion_data",
    "write_distortion_data",
    "read_distortion_data"
]

def write_distortion_data(file, data, validate=True, mode='npz'):
    if validate:
        data = validate_distortion_data(data)
    return write_tree(file, data, mode=mode)

def read_distortion_data(file, validate=True, mode='npz'):
    data = read_tree(file, mode=mode)
    if validate:
        data = validate_distortion_data(data)
    return data


schemas = {}
class DistortionDataValidationError(ValueError):
    ...

def validate_instance_type(value, type, validate_instances=False):
    if isinstance(type, str):
        if validate_instances:
            # try:
            value = validate_base_schema(type, schemas[type], value)
            # except DistortionDataValidationError:
            #     return False, value
            # else:
            return True, value
        else:
            return type, value
    elif isinstance(type, list):
        try:
            viter = iter(value)
        except TypeError:
            return False, value
        else:
            value = [
                validate_instance_type(v, type[0], validate_instances=True)
                for v in viter
            ]
            check = all(v[0] for v in value)
            value = [v[1] for v in value]
            return check, value
    elif not isinstance(type, tuple) and issubclass(type, np.ndarray):
        if isinstance(value, np.ndarray):
            return True, value
        else:
            try:
                value = np.asanyarray(value)
            except ValueError:
                return False, value
            else:
                return not any(
                    np.issubdtype(value.dtype, np.dtype(t))
                    for t in [bool, str, object]
                ), value
    else:
        return isinstance(value, type), value

def validate_base_schema(name, schema, instance):
    missing_keys = []
    for k in schema.get('$required', []):
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
    for k,v in instance.items():
        type = schema.get(k)
        if type is None:
            if not schema.get('$allow_extra_keys', False):
                extra_keys.append(k)
                continue
            else:
                type = schema.get('$default')
                if type is None: continue

        res, v = validate_instance_type(v, type)
        if isinstance(res, str):
            delayed_validations.append([k, v, res])
        elif not res:
            bad_types.append([k,v,type])
        else:
            instance[k] = v

    if len(extra_keys) > 0:
        raise DistortionDataValidationError(
            f"instance of schema `{name}` had excess keys {extra_keys}"
        )
    if len(bad_types) > 0:
        bad_types = "\n".join(f"{k}: expected {t} got {v}" for k,v,t in bad_types)
        raise DistortionDataValidationError(
            f"instance of schema `{name}` had mis-typed values {bad_types}"
        )
    if len(delayed_validations) > 0:
        for k,v,t in delayed_validations:
            instance[k] = validate_base_schema(t, schemas[t], v)

    validator = schema.get('$validator')
    if validator is not None:
        return validator(instance)
    else:
        return instance

def validate_distortion_data(instance, root='distortion_data'):
    return validate_base_schema(root, schemas[root], instance)

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

    return instance


schemas['structure'] = {
    'coordinates':np.ndarray,
    'energy':float,
    'hessian':np.ndarray,
    "$validator": validate_structure_schema,
    "$required": ['coordinates']
}

schemas['results'] = {
    'reactant':'structure',
    'transition_state':'structure',
}

class distortion_types:
    RigidSinglePoints = 'single_points'
    RelaxedSinglePoints = 'relaxed_single_points'
    MEPSampling = 'mep'

schemas['distortion'] = {
    'results':['results'],
    'spec':[(int, np.integer)],
    'distortion':np.ndarray,
    'magnitudes':np.ndarray,
    'gamma':float,
    'method':distortion_types,
    # '$validator':validate_distortion,
    '$required':['results']
}
def validate_solvothermal_distortion(instance):
    if len(instance['results']) != 1:
        raise DistortionDataValidationError(
            f"`solvothermal` case only takes one results (got {len(instance['results'])})"
        )
    if len(instance.keys()) > 1:
        raise DistortionDataValidationError(
            f"`solvothermal` only supports 'results' as keys"
        )

    for r in instance['results']:
        reactant = r['reactant']
        if 'hessian' not in reactant:
            raise DistortionDataValidationError(
                f"`solvothermal` requires `hessian` for `reactant` and `transition_state`"
            )
        transition_state = r['transition_state']
        if 'hessian' not in transition_state:
            raise DistortionDataValidationError(
                f"`solvothermal` requires `hessian` for `transition_state` (and `reactant`)"
            )

    return instance
def validate_internal_distortion(instance):
    if 'spec' not in instance:
        raise ValueError("internal requires a spec")

    return instance
def validate_direction_distortion(instance):
    if 'direction' not in instance:
        raise ValueError("force requires a direction")

    return instance

def validate_atom_counts(v, nats):
    for result in v['results']:
        for k in {'reactant', 'transition_state'}:
            c = result[k]['coordinates']
            if len(c) != nats:
                return False, k, len(c)
    else:
        return True, None, nats
def validate_lot(instance):
    if 'solvothermal' not in instance:
        raise DistortionDataValidationError("level of theory requires `solvothermal` results")
    nats = len(instance['atoms'])
    for k,v in instance.items():
        if k == "solvothermal":
            res, ck, c = validate_atom_counts(v, nats)
            if not res:
                raise DistortionDataValidationError(
                    f"`lot`: in key `{k}` mismatch between number of `atoms` ({nats}) and {ck} coordinates ({c})"
                )
            instance[k] = validate_solvothermal_distortion(v)
        elif k.startswith("internal_"):
            res, ck, c = validate_atom_counts(v, nats)
            if not res:
                raise DistortionDataValidationError(
                    f"`lot`: in key `{k}` mismatch between number of `atoms` ({nats}) and {ck} coordinates ({c})"
                )
            instance[k] = validate_internal_distortion(v)
        elif k.startswith("force_"):
            res, ck, c = validate_atom_counts(v, nats)
            if not res:
                raise DistortionDataValidationError(
                    f"`lot`: in key `{k}` mismatch between number of `atoms` ({nats}) and {ck} coordinates ({c})"
                )
            instance[k] = validate_direction_distortion(v)
        elif k in {'atoms', 'smiles'}:
            continue
        else:
            raise DistortionDataValidationError(
                "level of theory only supports `solvothermal`, `internal_{index}`, and `force_{index}` as keys"
            )

    return instance



schemas['lot'] = {
    'atoms':[str],
    'smiles':str,
    '$default':'distortion',
    '$allow_extra_keys':True,
    "$validator": validate_lot,
    "$required": ['atoms']
}

schemas['reactant_system'] = {
    '$default':'lot',
    '$allow_extra_keys':True
}

schemas['reaction_class'] = {
    '$default':'reactant_system',
    '$allow_extra_keys':True
}

schemas['distortion_data'] = {
    '$default':'reaction_class',
    '$allow_extra_keys':True
}