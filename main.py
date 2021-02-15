import sys


sys.path.insert(0, '.')

import biothings_resolver

r = biothings_resolver.ChemResolver()

failed_agents = {
    ('chem_CHEBI', 'INCHIKEY'),
    ('chem_CHEBI', 'DRUGBANK'),
}
print(r.resolve_path(
    'CHEBI', ['INCHIKEY'], failed_agents
))

sys.exit(0)
