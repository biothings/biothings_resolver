from .Resolver import Resolver
from . import BioThingsAPIAgent


class ChemResolver(Resolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FIXME: add these agents properly
        # {
        #   (PREFIX, field):
        #       PREFIX: {'f': field, 'cf': cost_fwd, 'cb': cost_backward} }
        # }
        DEFAULT_COST_FORWARD = 1.0
        DEFAULT_COST_BACKWARD = 1.5
        mychem_fields = {
            ('CHEBI', 'chebi.id'): {
                'INCHI': {'f': 'chebi.inchi'},
                'INCHIKEY': {'f': 'chebi.inchikey'},
                'DRUGBANK': {'f': 'chebi.xrefs.drugbank'},
                'HMDB': {'f': 'chebi.xrefs.hmdb'},
                # 'KEGG': {'f': 'chebi.xrefs.kegg_compound'},  # cmpnd or drug?
                'PUBCHEM.COMPOUND': {'f': 'chebi.xrefs.pubchem.cid'},
            },
            ('CHEMBL.COMPOUND', 'chembl.molecule_chembl_id'): {
                'INCHI': {'f': 'chembl.inchi', 'cb': 1.2},
                'INCHIKEY': {'f': 'chembl.inchi_key'},

            },
            ('DRUGBANK', 'drugbank.id'): {
                'INCHI': {'f': 'drugbank.inchi', 'cb': 1.1},
                'INCHIKEY': {'f': 'drugbank.inchi_key'},
                'CHEBI': {'f': 'drugbank.xrefs.chebi'},
                'CHEMBL.COMPOUND': {'f': 'drugbank.xrefs.chembl'},
                # 'KEGG': {'f': 'drugbank.xrefs.kegg.cid'},  # ditto
                'PHARMGKB.DRUG': {'f': 'drugbank.xrefs.pharmgkb'},
                'PUBCHEM.COMPOUND': {'f': 'drugbank.xrefs.pubchem.cid'}
            },
            ('PHARMGKB.DRUG', 'pharmgkb.id'): {
                'INCHI': {'f': 'pharmgkb.inchi'},
                'CHEBI': {'f': 'pharmgkb.xrefs.chebi'},
                'DRUGBANK': {'f': 'pharmgkb.xrefs.drugbank'},
                'HMDB': {'f': 'pharmgkb.xrefs.hmdb'},
                # 'KEGG': {'f': 'pharmgkb.xrefs.kegg_compound'},
                'MESH': {'f': 'pharmgkb.xrefs.mesh'},
                'PUBCHEM.COMPOUND': {'f': 'pharmgkb.xrefs.pubchem.cid'},
            },
            ('PUBCHEM.COMPOUND', 'pubchem.cid'): {
                'INCHI': {'f': 'pubchem.inchi'},
                'INCHIKEY': {'f': 'pubchem.inchi_key', 'cb': 1.0}
            },
            ('UNII', 'unii.unii'): {
                'INCHIKEY': {'f': 'unii.inchikey'},
            },
        }
        for (src_t, src_f), tgt_dict in mychem_fields.items():
            output_fields = {}
            agent_name = f'chem_{src_t}'
            for tgt_t, info in tgt_dict.items():
                tgt_f = info['f']
                output_fields.setdefault(tgt_t, []).append(tgt_f)
                cost_forward = info.get('cf', None)
                if cost_forward:
                    self.agent_costs[(agent_name, tgt_t)] = cost_forward
                cost_backward = info.get('cb', DEFAULT_COST_BACKWARD)
            agent = BioThingsAPIAgent('chem', src_t, [src_f], output_fields)
            self.agents[agent_name] = agent

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
        # self.agents.add('inchi', 'inchikey', BioThingsAPIAgent('chem', inchi_fields, inchikey_fields), cost=0.5)

        # we supply a default preference list
        self.preferred = ['INCHIKEY', 'UNII', 'DRUGBANK', 'CHEBI',
                          'CHEMBL.COMPOUND', 'PUBCHEM.COMPOUND']




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
