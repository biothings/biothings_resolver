# BioThings Resolver

Library to lookup BioThings API identifiers.

## Example Usage

### Creating a Resolver instance
```python
import biothings_resolver

# Using a pre-defined resolver
resolver = biothings_resolver.Resolver()

# Creating a user-defined resolver
resolver = biothings_resolver.Resolver()
# Setting identifier type preference
resolver.preferred = ['INCHIKEY', 'UNII']
```

### Adding resolver agents to a resolver
```python
import biothings_resolver

resolver = biothings_resolver.ChemResolver()
# Unfreeze the agents container to make updates
resolver.agents.frozen = False
resolver.agents.add(
    'CHEMBL.COMPOUND', 'INCHIKEY', biothings_resolver.BioThingsAPIAgent(
        'chem', 'chembl.molecule_chembl_id', 'chembl.inchi_key',
    ), cost=1.0             
)
# Re-freeze to update paths
resolver.agents.frozen = True
```

### Resolve CURIE style input
```python
import biothings_resolver

resolver = biothings_resolver.ChemResolver()
resolve_input = [
    "InChIKey:GKKDCARASOJPNG-UHFFFAOYSA-N",
    "[inchikey:DVARTQFDIMZBAA-UHFFFAOYSA-O]",  # safe_curie, not required
    "CHEBI:32146",
]
for orig, new in zip(resolve_input, resolver.resolve_curie(resolve_input, expand=True)):
    print("Original:", orig)
    for new_id in new:
        print(new_id)
    print()
```


### Using as a decorator
```python
import biothings_resolver

# For using as a decorator to update documents output from a function
resolver = biothings_resolver.ChemResolver([
        ('INCHI', 'pharmgkb.inchi'),
        ('PUBCHEM.COMPOUND', 'pharmgkb.xrefs.pubchem.cid'),
        ('DRUGBANK', 'pharmgkb.xrefs.drugbank'),
        ('CHEBI', 'pharmgkb.xrefs.chebi'),
])
@resolver
def load_data():
     # a typical dataloader for PharmGKB
    ... 
    for doc in ...:
        yield doc
```