import json
import numpy as np
from typing import List


def to_dict(obj, classkey: str = '__class__') -> dict:
    if isinstance(obj, dict):
        return {k: to_dict(v, classkey) for k, v in obj.items()}
    elif hasattr(obj, "_ast"):
        return to_dict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return [to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = {}
        for key, value in obj.__dict__.items():
            if not callable(value) and not key.startswith('_'):
                data[key] = to_dict(value, classkey)
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    return obj


def save(obj, path: str) -> None:
    if not path.endswith('.json'):
        path += '.json'
    with open(path, 'w') as f:
        json.dump(to_dict(obj, '__class__'), f, indent=2)


def from_dict(d, class_map: dict) -> dict:
    if isinstance(d, dict):
        if '__class__' in d:
            cls_name = d.pop('__class__')
            module_name = d.pop('__module__', None)
            if cls_name in class_map:
                cls = class_map[cls_name]
            else:
                if module_name:
                    module = __import__(module_name, fromlist=[cls_name])
                    cls = getattr(module, cls_name)
                else:
                    cls = globals().get(cls_name)
            instance = cls.__new__(cls)
            for key, value in d.items():
                setattr(instance, key, from_dict(value, class_map))
            return instance
        return {k: from_dict(v, class_map) for k, v in d.items()}
    elif isinstance(d, list):
        return [from_dict(v, class_map) for v in d]
    else:
        return d


def load(path: str, class_map: dict):
    if not path.endswith('json'):
        raise ValueError(f"Provided path '{path}' is not a JSON file.")
    with open(path , 'r') as f:
        obj = from_dict(json.load(f), class_map)
    return obj


import math
from typing import Any, List
try:
    import numpy as np
except Exception:
    np = None

# optional: better visual width for unicode
try:
    from wcwidth import wcswidth
except Exception:
    wcswidth = None

def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))

def _w(s: Any) -> int:
    s = "" if s is None else str(s)
    return wcswidth(s) if (wcswidth is not None) else len(s)
