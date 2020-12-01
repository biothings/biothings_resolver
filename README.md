# BioThings Resolver

Library to lookup BioThings API identifiers.

## Installation

To install, clone the project and install it, or run the following command.
It will be released as a package in the future when it is more mature.

```shell script
pip install git+https://github.com/biothings/biothings_resolver.git
```

## Example Usage

### Using Predefined Resolvers

Predefined Resolvers can be imported from `biothings_resolver.predefined_resolvers`.
Currently, the following are implemented:

- `ChemResolver` which resolves drug identifiers available in MyChem.info
- `GeneResolver` which resolves gene identifiers available in MyGene.info

#### Supported input and output identifier types

Check by running:
```python
In [1]: import biothings_resolver 
   ...: resolver = biothings_resolver.predefined_resolvers.GeneResolver() 
   ...:  
   ...: print("Inputs:", ", ".join(resolver.agents.sources)) 
   ...: print("Outputs:", ", ".join(resolver.agents.targets))                   
Inputs: WormBase, ZFIN, FlyBase, NCBIGene, HGNC, MGI, ENSEMBL, UniProtKB, RGD
Outputs: WormBase, ZFIN, FlyBase, NCBIGene, HGNC, MGI, ENSEMBL, UniProtKB, RGD
```

#### Resolve identifiers

Takes in a sequence of dictionaries, returns a list of dictionaries in the same
order, the items in the returned dictionary contains lists of new identifiers.

It can either return the most preferred available identifier type, or expand to
have all available preferred types, as defined in `Resolver.preferred`. 

```python
In [1]: import biothings_resolver 
   ...:  
   ...: resolver = biothings_resolver.predefined_resolvers.ChemResolver() 
   ...: resolve_input = [{ 
   ...:     'INCHI': 'InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3', 
   ...:     'DRUGBANK': 'DB00201', 
   ...: }]                                                                      

In [2]: # don't expand 
   ...: resolver.expand = False
   ...: output = list(resolver.resolve_identifier(resolve_input)) 
   ...: for k, identifiers in output[0].items(): 
   ...:     print(f"{k}:", ", ".join(identifiers)) 
   ...:                                                                         
INCHIKEY: RYYVLZVUVIJVGH-UHFFFAOYSA-N

In [3]: # or expand 
   ...: resolver.expand = True
   ...: output = list(resolver.resolve_identifier(resolve_input)) 
   ...: for k, identifiers in output[0].items(): 
   ...:     print(f"{k}:", ", ".join(identifiers)) 
   ...:                                                                         
INCHIKEY: RYYVLZVUVIJVGH-UHFFFAOYSA-N
UNII: 3G6A5W338E
DRUGBANK: DB00201
CHEBI: CHEBI:27732
CHEMBL.COMPOUND: CHEMBL113
PUBCHEM.COMPOUND: 2519
```

#### Resolve CURIE style input, expanding to obtain all available fields

Using a shorter CURIE-like input, is more compact than having a dictionary
input. This is shorter but one input can only have one identifier, where the
previous method can support multiple inputs for a single query, providing the
resolver with more information.

```python
In [1]: import biothings_resolver 
   ...:  
   ...: resolver = biothings_resolver.predefined_resolvers.ChemResolver() 
   ...: resolver.expand = True
   ...: resolve_input = [ 
   ...:     "InChIKey:GKKDCARASOJPNG-UHFFFAOYSA-N", 
   ...:     "[inchikey:DVARTQFDIMZBAA-UHFFFAOYSA-O]",  # safe_curie, not required 
   ...:     "CHEBI:32146", 
   ...: ] 
   ...: for orig, new in zip(resolve_input, resolver.resolve_curie(resolve_input)): 
   ...:     print("Original:", orig) 
   ...:     for new_id in new: 
   ...:         print(new_id) 
   ...:     print() 
   ...:                                                                                                                                   
Original: InChIKey:GKKDCARASOJPNG-UHFFFAOYSA-N
INCHIKEY:GKKDCARASOJPNG-UHFFFAOYSA-N
CHEBI:81931
CHEMBL.COMPOUND:CHEMBL2251334
PUBCHEM.COMPOUND:61021

Original: [inchikey:DVARTQFDIMZBAA-UHFFFAOYSA-O]
INCHIKEY:DVARTQFDIMZBAA-UHFFFAOYSA-O
UNII:T8YA51M7Y6
CHEBI:63038
CHEMBL.COMPOUND:CHEMBL1500032
PUBCHEM.COMPOUND:22985

Original: CHEBI:32146
INCHIKEY:SUKJFIGYRHOWBL-UHFFFAOYSA-N
UNII:DY38VHM5OD
DRUGBANK:DBSALT001517
CHEBI:32146
CHEMBL.COMPOUND:CHEMBL1334078
PUBCHEM.COMPOUND:23665760
```

#### Use as a decorator

When using as a decorator, the decorated function/method must return/yield
dictionaries, then resolver will update a specific field in the dictionary to
contain an identifier in the order of preference.

The field key is specified in `Resolver.document_resolve_id_field` and by
default it is set to `_id`.

```python
import biothings_resolver

# For using as a decorator to update documents output from a function
resolver = biothings_resolver.predefined_resolvers.ChemResolver([
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

#### Change Pathfinding Options

Pathfinding behavior can be configured by altering attributes of the `Resolver`
instance. Currently the following can be modified.

##### Maximum path length

The maximum number of agents that `Resolver` will use in one run. It can be set
by changing the `max_path_length` attribute. Default is 3.

```python
import biothings_resolver

resolver = biothings_resolver.predefined_resolvers.ChemResolver()
resolver.max_path_length = 5
```

### Using customized `Resolver` instance

#### 1. Initialize an empty `Resolver`
```python
import biothings_resolver

# Using a pre-defined resolver
resolver = biothings_resolver.Resolver()

# Setting identifier type preference
resolver.preferred = ['INCHIKEY', 'UNII']
```

#### 2. Add resolver agents to a resolver
```python
import biothings_resolver

resolver = biothings_resolver.Resolver()
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

#### 3. Use
