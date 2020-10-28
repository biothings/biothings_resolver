import os
import yaml


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


classes, prefixes = parse_classes_prefixes()
