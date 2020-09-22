# BioThings IDLookup

Library to lookup BioThings API identifiers.

## Example Usage

```python
import biothings_idlookup

lookup = biothings_idlookup.IDLookupMyChem([
        ('inchi', 'pharmgkb.inchi'),
        ('pubchem', 'pharmgkb.xrefs.pubchem.cid'),
        ('drugbank', 'pharmgkb.xrefs.drugbank'),
        ('chebi', 'pharmgkb.xrefs.chebi'),
])
    
@lookup
def load_data():
     # a typical dataloader for PharmGKB
    ... 
    for doc in ...:
        yield doc
```