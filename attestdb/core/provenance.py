"""Source URL resolution — turn source_id strings into clickable URLs."""

from __future__ import annotations


def resolve_source_url(source_id: str, source_type: str = "") -> str | None:
    """Resolve a claim's source_id to a URL pointing to the original source.

    Works retroactively on all existing claims — no re-ingestion needed.
    Returns None when no URL can be derived (aggregated KGs, missing info).
    """
    if not source_id:
        return None

    parts = source_id.split(":")

    # --- Bibliographic / PubMed ---
    if source_id.startswith("semmeddb:") and len(parts) >= 2:
        pmid = parts[1]
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

    # GWAS: gwas:{snp}:{pmid} or gwas:{snp}
    if source_id.startswith("gwas:") and len(parts) >= 2:
        if len(parts) >= 3:
            pmid = parts[2]
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        snp = parts[1]
        return f"https://www.ebi.ac.uk/gwas/variants/{snp}"

    # --- Protein / Gene databases ---
    if source_id.startswith("goa:") and len(parts) >= 2:
        uniprot = parts[1]
        return f"https://www.uniprot.org/uniprot/{uniprot}"

    if source_id.startswith("uniprot:") and len(parts) >= 2:
        accession = parts[1]
        return f"https://www.uniprot.org/uniprot/{accession}"

    # CTD: ctd:cg:{mesh}:{gene}, ctd:gd:{gene}:{disease}, ctd:cd:{mesh}:{disease}
    if source_id.startswith("ctd:") and len(parts) >= 4:
        subtype = parts[1]
        if subtype == "cg":
            gene = parts[3]
            return f"https://ctdbase.org/detail.go?type=gene&acc={gene}"
        if subtype == "gd":
            gene = parts[2]
            return f"https://ctdbase.org/detail.go?type=gene&acc={gene}"
        if subtype == "cd":
            mesh = parts[2]
            return f"https://ctdbase.org/detail.go?type=chem&acc={mesh}"

    if source_id.startswith("biogrid:") and len(parts) >= 2:
        entity_a = parts[1]
        return f"https://thebiogrid.org/search.php?search={entity_a}"

    if source_id.startswith("intact:") and len(parts) >= 2:
        interaction = parts[1]
        return f"https://www.ebi.ac.uk/intact/details/interaction/{interaction}"

    if source_id.startswith("pharmgkb:") and len(parts) >= 2:
        id1 = parts[1]
        return f"https://www.pharmgkb.org/search/{id1}"

    if source_id.startswith("opentargets:") and len(parts) >= 2:
        target = parts[1]
        return f"https://platform.opentargets.org/target/{target}"

    if source_id.startswith("hpo:") and len(parts) >= 3:
        hpo_term = parts[2]
        return f"https://hpo.jax.org/browse/term/{hpo_term}"

    if source_id.startswith("sider:") and len(parts) >= 2:
        stitch = parts[1]
        return f"http://sideeffects.embl.de/drugs/{stitch}"

    # --- Ontologies ---
    if source_id.startswith("doid:") and len(parts) >= 2:
        doid = parts[1]
        return f"https://disease-ontology.org/term/DOID:{doid}"

    if source_id.startswith("chebi:") and len(parts) >= 2:
        chebi_id = parts[1]
        return f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{chebi_id}"

    if source_id.startswith("mesh:") and len(parts) >= 2:
        ui = parts[1]
        return f"https://meshb.nlm.nih.gov/record/ui?ui={ui}"

    if source_id.startswith("mondo:") and len(parts) >= 2:
        mondo_id = parts[1]
        return f"https://monarchinitiative.org/disease/MONDO:{mondo_id}"

    # --- Clinical ---
    if source_id.startswith("ct:") and len(parts) >= 2:
        nct = parts[1]
        return f"https://clinicaltrials.gov/study/{nct}"

    if source_id.startswith("omim:") and len(parts) >= 2:
        mim = parts[1]
        return f"https://omim.org/entry/{mim}"

    if source_id.startswith("bindingdb:") and len(parts) >= 2:
        uniprot = parts[1]
        return f"https://www.bindingdb.org/uniprot/{uniprot}"

    if source_id.startswith("dgidb:") and len(parts) >= 2:
        gene = parts[1]
        return f"https://dgidb.org/genes/{gene}"

    # --- Chat / collaboration ---
    if source_id.startswith("slack:") and len(parts) >= 3:
        channel = parts[1]
        ts = parts[2].replace(".", "")
        return f"https://slack.com/archives/{channel}/p{ts}"

    if source_id.startswith("gmail:") and len(parts) >= 2:
        thread = parts[1]
        return f"https://mail.google.com/mail/u/0/#inbox/{thread}"

    # github:{repo}#{num} — colon-separated repo, # for issue/PR number
    if source_id.startswith("github:"):
        rest = source_id[len("github:"):]
        if "#" in rest:
            repo, num = rest.rsplit("#", 1)
            return f"https://github.com/{repo}/issues/{num}"

    if source_id.startswith("gdocs:") and len(parts) >= 2:
        doc_id = parts[1]
        return f"https://docs.google.com/document/d/{doc_id}"

    if source_id.startswith("notion:") and len(parts) >= 2:
        page_id = parts[1]
        return f"https://notion.so/{page_id}"

    # --- Aggregated KGs (dataset-level URLs) ---
    if source_id.startswith("hetionet:"):
        return "https://het.io"
    if source_id.startswith("kg2c:"):
        return "https://github.com/RTXteam/RTX-KG2"
    if source_id.startswith("monarch:"):
        return "https://monarchinitiative.org"
    if source_id.startswith("pharmebinet:"):
        return "https://github.com/rephetio/PharMeBINet"

    # Dataset-level: match full source_id or source_type
    _DATASET_URLS = {
        "string_ppi": "https://string-db.org",
        "reactome": "https://reactome.org",
        "disgenet_curated": "https://disgenet.org",
    }
    if source_id in _DATASET_URLS:
        return _DATASET_URLS[source_id]
    if source_type in _DATASET_URLS:
        return _DATASET_URLS[source_type]

    # --- No URL derivable (primekg, drkg, jira without base_url, unknown) ---
    return None
