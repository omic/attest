"""Gene ID mapper for cross-source entity resolution.

Maps Ensembl protein IDs (ENSP) and gene symbols to Hetionet-style
Entrez Gene IDs (Gene_XXXX). Used by DISEASES, STRING, and other loaders.
Also provides reverse mapping (Entrez → gene symbol) for display name resolution.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os

logger = logging.getLogger(__name__)

# NCBI gene2ensembl mapping file (human, ~30MB compressed)
GENE2ENSEMBL_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz"
# NCBI gene_info for comprehensive symbol coverage (~40MB compressed, ~65K human genes)
GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
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
        self._uniprot_to_entrez: dict[str, int] = {}

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

        # Parse: columns are tax_id, GeneID, Ensembl_gene,
        # RNA_nuc_acc, Ensembl_rna, protein_acc, Ensembl_prot
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

    def entrez_to_symbol(self, entrez_id: int) -> str | None:
        """Map an Entrez Gene ID to its gene symbol.

        Returns the gene symbol (e.g., 'CDK4') or None if unknown.
        Coverage depends on data loaded: Hetionet (~20K genes) and/or
        NCBI gene_info (~65K human genes).
        """
        return self._entrez_to_symbol.get(entrez_id)

    def load_gene_info(self) -> None:
        """Download NCBI gene_info for comprehensive Entrez→symbol coverage.

        Extends _entrez_to_symbol and _symbol_to_entrez with ~65K human genes
        (vs Hetionet's ~20K). Covers genes from CTD, ClinVar, HPO, etc.
        that aren't in Hetionet.
        """
        import requests

        dest = os.path.join(self._cache_dir, "Homo_sapiens.gene_info.gz")
        if not os.path.exists(dest):
            logger.info("Downloading NCBI gene_info (~40MB)...")
            resp = requests.get(GENE_INFO_URL, timeout=120, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Downloaded gene_info.gz (%d bytes)", os.path.getsize(dest))

        # Format: tax_id(0) GeneID(1) Symbol(2) ... (15+ columns)
        count = 0
        with gzip.open(dest, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                if parts[0] != HUMAN_TAX_ID:
                    continue
                try:
                    entrez_id = int(parts[1])
                except ValueError:
                    continue
                symbol = parts[2].strip()
                if symbol and symbol != "-":
                    # Don't overwrite existing mappings from Hetionet
                    if entrez_id not in self._entrez_to_symbol:
                        self._entrez_to_symbol[entrez_id] = symbol
                        count += 1
                    if symbol.upper() not in self._symbol_to_entrez:
                        self._symbol_to_entrez[symbol.upper()] = entrez_id
        logger.info(
            "NCBI gene_info: %d new Entrez→symbol mappings (total: %d)",
            count, len(self._entrez_to_symbol),
        )

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

    def uniprot_to_gene_entity_id(self, uniprot_id: str) -> str | None:
        """Map a UniProt accession to a Hetionet Gene entity ID.

        'P24941' -> 'Gene_1017' or None
        """
        entrez = self._uniprot_to_entrez.get(uniprot_id)
        if entrez is not None:
            return f"Gene_{entrez}"
        return None

    def load_uniprot_mapping(self, path: str | None = None) -> None:
        """Load UniProt-to-Entrez mapping from idmapping_selected.tab.gz.

        Downloads HUMAN_9606_idmapping_selected.tab.gz from UniProt FTP if
        not cached. Only needs columns 1 (UniProt-AC) and 3 (GeneID).
        """
        import requests

        if path is None:
            path = os.path.join(self._cache_dir, "HUMAN_9606_idmapping_selected.tab.gz")

        if not os.path.exists(path):
            url = (
                "https://ftp.uniprot.org/pub/databases/uniprot/"
                "current_release/knowledgebase/idmapping/by_organism/"
                "HUMAN_9606_idmapping_selected.tab.gz"
            )
            logger.info("Downloading UniProt ID mapping (~100MB)...")
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Downloaded UniProt mapping to %s", path)

        count = 0
        with gzip.open(path, "rt") as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                uniprot_ac = parts[0].strip()
                gene_id_str = parts[2].strip()
                if uniprot_ac and gene_id_str:
                    # gene_id_str can be semicolon-separated; take first
                    for gid in gene_id_str.split(";"):
                        gid = gid.strip()
                        if gid:
                            try:
                                self._uniprot_to_entrez[uniprot_ac] = int(gid)
                                count += 1
                            except ValueError:
                                pass
                            break
        logger.info("UniProt mapping: %d accessions -> Entrez", count)

