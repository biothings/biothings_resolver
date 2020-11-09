import os
import yaml

from .containers import CanonDict


def get_classes_prefixes():
    path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(path, 'biolink-model.yaml')) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
        for cls, item in d['classes'].items():
            if 'id_prefixes' in item:
                for prefix in item['id_prefixes']:
                    yield cls, prefix


def parse_classes_prefixes():
    classes = {}
    prefixes = {}
    for cls, prefix in get_classes_prefixes():
        classes.setdefault(cls, set()).add(prefix)
        prefixes.setdefault(prefix, set()).add(cls)
    return classes, prefixes


def canonical_prefixes(raw_prefixes: dict):
    canon_maps = {
        'FB': 'FlyBase',
        'WB': 'WormBase',
    }
    remove_prefixes = [
        'DBSNP'
    ]
    d = CanonDict()
    d.case_sensitive = True  # will allow checking later when toggling this
    for k, v in raw_prefixes.items():
        if k in canon_maps or k in remove_prefixes:
            continue
        d[k] = v
    for k, v in canon_maps.items():
        d.add_alias(k, v)
    d.case_sensitive = False
    return d
