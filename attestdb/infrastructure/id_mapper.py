"""Gene ID mapper for cross-source entity resolution.

Maps Ensembl protein IDs (ENSP) and gene symbols to Hetionet-style
Entrez Gene IDs (Gene_XXXX). Used by DISEASES and STRING loaders.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os

logger = logging.getLogger(__name__)

# NCBI gene2ensembl mapping file (human, ~30MB compressed)
GENE2ENSEMBL_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz"
HUMAN_TAX_ID = "9606"


class GeneIDMapper:
    """Maps Ensembl protein IDs and gene symbols to Entrez Gene IDs.

    Downloads NCBI gene2ensembl.gz and parses Hetionet nodes to build
    lookup tables for cross-source entity resolution.
    """

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        self._ensp_to_entrez: dict[str, int] = {}
        self._symbol_to_entrez: dict[str, int] = {}
        self._entrez_to_symbol: dict[int, str] = {}

    def load(self, hetionet_nodes_path: str | None = None) -> None:
        """Download gene2ensembl.gz and parse Hetionet nodes. Build lookup tables.

        Args:
            hetionet_nodes_path: Path to hetionet-nodes.tsv. If provided,
                builds symbol_to_entrez fallback mapping.
        """
        self._load_gene2ensembl()
        if hetionet_nodes_path:
            self._load_hetionet_symbols(hetionet_nodes_path)
        logger.info(
            "GeneIDMapper loaded: %d ENSP->Entrez, %d symbol->Entrez",
            len(self._ensp_to_entrez), len(self._symbol_to_entrez),
        )

    def _load_gene2ensembl(self) -> None:
        """Download and parse NCBI gene2ensembl.gz for human genes."""
        import requests

        dest = os.path.join(self._cache_dir, "gene2ensembl.gz")
        if not os.path.exists(dest):
            logger.info("Downloading gene2ensembl.gz from NCBI...")
            resp = requests.get(GENE2ENSEMBL_URL, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            logger.info("Downloaded gene2ensembl.gz (%d bytes)", len(resp.content))

        # Parse: columns are tax_id, GeneID, Ensembl_gene, RNA_nuc_acc, Ensembl_rna, protein_acc, Ensembl_prot
        with gzip.open(dest, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 7:
                    continue
                tax_id = row[0]
                if tax_id != HUMAN_TAX_ID:
                    continue
                try:
                    entrez_id = int(row[1])
                except (ValueError, IndexError):
                    continue
                # Ensembl protein ID (column 6, 0-indexed)
                ensp = row[6].strip()
                if ensp and ensp != "-":
                    # Strip version suffix (ENSP00000251968.2 -> ENSP00000251968)
                    ensp_base = ensp.split(".")[0]
                    self._ensp_to_entrez[ensp_base] = entrez_id

    def _load_hetionet_symbols(self, nodes_path: str) -> None:
        """Parse Hetionet nodes TSV to build symbol->entrez mapping.

        Hetionet nodes format: id<TAB>name
        Gene nodes look like: Gene::9021<TAB>CDK4
        """
        with open(nodes_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 2:
                    continue
                raw_id = row[0].strip()
                name = row[1].strip()
                if raw_id.startswith("Gene::"):
                    try:
                        entrez_id = int(raw_id.split("::", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    # Map gene symbol -> entrez
                    if name:
                        self._symbol_to_entrez[name.upper()] = entrez_id
                        self._entrez_to_symbol[entrez_id] = name

    def ensp_to_gene_entity_id(self, ensp: str) -> str | None:
        """Map an ENSP ID to a Hetionet Gene entity ID.

        'ENSP00000251968' -> 'Gene_9021' or None
        """
        ensp_base = ensp.split(".")[0]
        entrez = self._ensp_to_entrez.get(ensp_base)
        if entrez is not None:
            return f"Gene_{entrez}"
        return None

    def string_protein_to_gene_entity_id(self, string_id: str) -> str | None:
        """Map a STRING protein ID to a Hetionet Gene entity ID.

        '9606.ENSP00000251968' -> strip prefix -> lookup -> 'Gene_9021'
        """
        # STRING format: {tax_id}.{ENSP_id}
        if "." in string_id:
            ensp = string_id.split(".", 1)[1]
        else:
            ensp = string_id
        return self.ensp_to_gene_entity_id(ensp)

    def symbol_to_gene_entity_id(self, symbol: str) -> str | None:
        """Map a gene symbol to a Hetionet Gene entity ID.

        'CDK4' -> 'Gene_9021' or None
        """
        entrez = self._symbol_to_entrez.get(symbol.upper())
        if entrez is not None:
            return f"Gene_{entrez}"
        return None

