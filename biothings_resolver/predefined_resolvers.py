from .Resolver import Resolver
from . import BioThingsAPIAgent


def _bt_chem(scope, fields):
    return BioThingsAPIAgent('chem', scope, fields)


def _bt_gene(scope, fields):
    return BioThingsAPIAgent('gene', scope, fields)


class ChemResolver(Resolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FIXME: add these agents properly

        self.agents.add('DRUGBANK', 'PUBCHEM.COMPOUND',
                        BioThingsAPIAgent('chem', ['drugbank.id'], ['pubchem.cid']))
        self.agents.add('PHARMGKB.DRUG', 'DRUGBANK',
                      BioThingsAPIAgent('chem', ['pharmgkb.id'], ['pharmgkb.xrefs.drugbank']))
        self.agents.add('inchi', 'PUBCHEM.COMPOUND',
                        BioThingsAPIAgent('chem', ['pubchem.inchi'], ['pubchem.cid']), cost=1.0)
        self.agents.add('inchi', 'drugbank', BioThingsAPIAgent('chem', ['drugbank.inchi'], ['drugbank.id']),
                      cost=1.1)
        self.agents.add('inchi', 'CHEMBL.COMPOUND', BioThingsAPIAgent('chem', 'chembl.inchi', 'chembl.molecule_chembl_id'),
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
        self.agents.add('PUBCHEM.COMPOUND', 'inchikey', BioThingsAPIAgent('chem', 'pubchem.cid', inchikey_fields))
        self.agents.add('drugbank', 'inchikey', BioThingsAPIAgent('chem', 'drugbank.id', inchikey_fields))
        self.agents.add('CHEMBL.COMPOUND', 'inchikey', BioThingsAPIAgent('chem', 'chembl.molecule_chembl_id', inchikey_fields))

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
            'PHARMGKB.DRUG', 'PUBCHEM.COMPOUND',
            BioThingsAPIAgent('chem', 'pharmgkb.id',
                              'pharmgkb.xrefs.pubchem.cid'),
            cost=1000.0
        )
        self.agents.add(
            'PUBCHEM.COMPOUND', 'PHARMGKB.DRUG',
            BioThingsAPIAgent('chem', 'pharmgkb.xrefs.pubchem.cid',
                              'pharmgkb.id'),
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
        self.agents.add(
            'CHEBI', 'INCHIKEY',
            BioThingsAPIAgent('chem', 'chebi.id', 'chebi.inchikey'),
            cost=1000.0
        )

        # we supply a default preference list
        self.preferred = ['INCHIKEY', 'UNII', 'DRUGBANK', 'CHEBI',
                          'CHEMBL.COMPOUND', 'PUBCHEM.COMPOUND', 'drugname']

        self.agents.frozen = True

        self.curie_in_xfrm['CHEBI'] = lambda x: x
        self.curie_out_xfrm['CHEBI'] = lambda x: x.replace('CHEBI:', '', 1)


class GeneResolver(Resolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agents.add('ENSEMBL', 'NCBIGene',
                        _bt_gene('ensembl.gene', 'entrezgene'))
        self.agents.add('NCBIGene', 'ENSEMBL',
                        _bt_gene('entrezgene', 'ensembl.gene'))
        field_mappings = [
            ('FlyBase', 'FLYBASE'),
            ('HGNC', 'HGNC'),
            ('MGI', 'MGI'),
            ('RGD', 'RGD'),
            ('WormBase', 'WormBase'),
            ('ZFIN', 'ZFIN'),
            ('UniProtKB', 'uniprot.Swiss-Prot'),
            # ('SGD', 'pantherdb.SGD'),  # Cost = 1.0?
            # ('PomBase', 'pantherdb.PomBase'),
            # ('dictyBase', 'pantherdb.dictyBase'),
        ]
        for src_t, src_scope in field_mappings:
            self.agents.add(src_t, 'NCBIGene',
                            _bt_gene(src_scope, 'entrezgene'))
            self.agents.add(src_t, 'ENSEMBL',
                            _bt_gene(src_scope, 'ensembl.gene'))
            self.agents.add('NCBIGene', src_t,
                            _bt_gene('entrezgene', src_scope))
            self.agents.add('ENSEMBL', src_t,
                            _bt_gene('ensembl.gene', src_scope))

        self.agents.frozen = True

        self.preferred = ['NCBIGene', 'ENSEMBL']
