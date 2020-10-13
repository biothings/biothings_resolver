from .IDLookup import IDLookup
from . import BioThingsAPIAgent


class IDLookupMyChem(IDLookup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agents.add('drugbank', 'pubchem', BioThingsAPIAgent('chem', ['drugbank.id'], ['pubchem.cid']))
        self.agents.add('pharmgkb', 'drugbank',
                      BioThingsAPIAgent('chem', ['pharmgkb.id'], ['pharmgkb.xrefs.drugbank']))
        self.agents.add('inchi', 'pubchem', BioThingsAPIAgent('chem', ['pubchem.inchi'], ['pubchem.cid']), cost=1.0)
        self.agents.add('inchi', 'drugbank', BioThingsAPIAgent('chem', ['drugbank.inchi'], ['drugbank.id']),
                      cost=1.1)
        self.agents.add('inchi', 'chembl', BioThingsAPIAgent('chem', 'chembl.inchi', 'chembl.molecule_chembl_id'),
                      cost=1.2)

        inchi_fields = [
            'pubchem.inchi',
            'drugbank.inchi',
            'chembl.inchi'
        ]
        inchikey_fields = [
            'pubchem.inchi_key',
            'drugbank.inchi_key',
            'chembl.inchi_key'
        ]

        # inchi to inchikey (direct route)
        self.agents.add('inchi', 'inchikey', BioThingsAPIAgent('chem', inchi_fields, inchikey_fields), cost=0.5)
        # indirect route
        self.agents.add('pubchem', 'inchikey', BioThingsAPIAgent('chem', 'pubchem.cid', inchikey_fields))
        self.agents.add('drugbank', 'inchikey', BioThingsAPIAgent('chem', 'drugbank.id', inchikey_fields))
        self.agents.add('chembl', 'inchikey', BioThingsAPIAgent('chem', 'chembl.molecule_chembl_id', inchikey_fields))

        # these agents have not been verified
        self.agents.add(
            'unii', 'inchikey',
            BioThingsAPIAgent('chem', 'unii.unii', 'unii.inchikey'),
            cost=1000.0
        )
        self.agents.add(
            'inchikey', 'unii',
            BioThingsAPIAgent('chem', 'unii.inchikey', 'unii.unii'),
            cost=1000.0
        )
        self.agents.add(
            'pharmgkb', 'pubchem',
            BioThingsAPIAgent('chem', 'pharmgkb.id',
                              'pharmgkb.xrefs.pubchem.cid'),
            cost=1000.0
        )
        self.agents.add(
            'pubchem', 'pharmgkb',
            BioThingsAPIAgent('chem', 'pharmgkb.xrefs.pubchem.cid',
                              'pharmgkb.id'),
            cost=1000.0
        )
        self.agents.add(
            'pharmgkb', 'rxnorm',
            BioThingsAPIAgent('chem', 'pharmgkb.id',
                              'pharmgkb.xrefs.rxnorm'),
            cost=1000.0
        )
        self.agents.add(
            'inchikey', 'drugbank',
            BioThingsAPIAgent('chem', 'drugbank.inchi_key', 'drugbank.id'),
            cost=1000.0
        )
        self.agents.add(
            'inchikey', 'chebi',
            BioThingsAPIAgent('chem', 'chebi.inchikey', 'chebi.id'),
            cost=1000.0
        )
        self.agents.add(
            'inchikey', 'inchi',
            BioThingsAPIAgent('chem', inchikey_fields, inchi_fields),
            cost=2000.0
        )




        # we supply a default preference list
        self.preferred = ['inchikey', 'unii', 'rxnorm', 'drugbank', 'chebi',
                          'chembl', 'pubchem', 'drugname']

        self.agents.frozen = True
