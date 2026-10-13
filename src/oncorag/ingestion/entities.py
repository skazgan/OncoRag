"""Lightweight, dictionary-based oncology entity extraction (genes, cancer types, variants).

Why not a neural NER model? For retrieval *metadata* we want something deterministic, fast,
auditable and with zero model downloads. The lexicons are deliberately small and curated; the
extractor is an interface, so a PubTator / scispaCy backend can replace it without touching the
rest of the pipeline.

Matching rules that matter in practice:
* Gene symbols are matched **case-sensitively** (``MET`` the gene vs. "met" the verb).
* Longest match wins and matches never overlap, so "non-small cell lung cancer" is not also
  reported as "small cell lung cancer".
* Acronyms for diseases are case-sensitive; spelled-out names are case-insensitive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# canonical HGNC-style symbol -> aliases written in the literature
GENES: dict[str, list[str]] = {
    "TP53": ["p53"],
    "KRAS": [],
    "NRAS": [],
    "HRAS": [],
    "BRAF": [],
    "EGFR": ["HER1"],
    "ERBB2": ["HER2", "HER-2", "HER2/neu"],
    "ERBB3": ["HER3"],
    "MET": ["c-MET", "c-Met"],
    "ALK": [],
    "ROS1": [],
    "RET": [],
    "NTRK1": [],
    "NTRK2": [],
    "NTRK3": [],
    "PIK3CA": [],
    "PTEN": [],
    "AKT1": [],
    "MTOR": ["mTOR"],
    "APC": [],
    "CTNNB1": ["beta-catenin"],
    "SMAD4": [],
    "BRCA1": [],
    "BRCA2": [],
    "PALB2": [],
    "ATM": [],
    "ATR": [],
    "CHEK2": [],
    "MLH1": [],
    "MSH2": [],
    "MSH6": [],
    "PMS2": [],
    "POLE": [],
    "CDKN2A": ["p16INK4a", "p16"],
    "CDK4": [],
    "CDK6": [],
    "RB1": [],
    "CCND1": [],
    "MYC": ["c-MYC", "c-Myc"],
    "MYCN": ["N-MYC", "N-Myc"],
    "IDH1": [],
    "IDH2": [],
    "FGFR1": [],
    "FGFR2": [],
    "FGFR3": [],
    "VHL": [],
    "NF1": [],
    "STK11": ["LKB1"],
    "KEAP1": [],
    "ARID1A": [],
    "SMARCA4": ["BRG1"],
    "NOTCH1": [],
    "FBXW7": [],
    "TERT": [],
    "JAK2": [],
    "KIT": [],
    "PDGFRA": [],
    "FLT3": [],
    "NPM1": [],
    "DNMT3A": [],
    "TET2": [],
    "EZH2": [],
    "BCL2": [],
    "ABL1": ["BCR-ABL", "BCR-ABL1", "BCR::ABL1"],
    "CD274": ["PD-L1", "PDL1"],
    "PDCD1": ["PD-1"],
    "CTLA4": ["CTLA-4"],
    "ESR1": [],
    "MDM2": [],
    "SF3B1": [],
    "SPOP": [],
    "VEGFA": ["VEGF"],
    "TMPRSS2": [],
    "ERG": [],
    "CDH1": [],
    "GATA3": [],
    "FOXA1": [],
    "KMT2D": [],
    "ASXL1": [],
    "RUNX1": [],
    "CEBPA": [],
    "SETD2": [],
    "BAP1": [],
    "PBRM1": [],
    "CIC": [],
    "H3F3A": [],
}

# canonical name -> aliases. Acronyms (all caps) are matched case-sensitively.
CANCER_TYPES: dict[str, list[str]] = {
    "non-small cell lung cancer": [
        "NSCLC",
        "non-small-cell lung cancer",
        "non small cell lung cancer",
        "non-small cell lung carcinoma",
        "lung adenocarcinoma",
        "LUAD",
        "lung squamous cell carcinoma",
        "LUSC",
    ],
    "small cell lung cancer": ["SCLC", "small-cell lung cancer", "small cell lung carcinoma"],
    "lung cancer": ["lung carcinoma"],
    "breast cancer": ["breast carcinoma", "mammary carcinoma"],
    "triple-negative breast cancer": ["TNBC", "triple negative breast cancer"],
    "colorectal cancer": [
        "CRC",
        "colon cancer",
        "rectal cancer",
        "colorectal carcinoma",
        "colorectal adenocarcinoma",
        "colon adenocarcinoma",
    ],
    "melanoma": ["cutaneous melanoma", "uveal melanoma"],
    "pancreatic cancer": [
        "PDAC",
        "pancreatic ductal adenocarcinoma",
        "pancreatic adenocarcinoma",
    ],
    "prostate cancer": ["prostate carcinoma", "prostate adenocarcinoma", "CRPC"],
    "ovarian cancer": ["ovarian carcinoma", "high-grade serous ovarian cancer", "HGSOC"],
    "gastric cancer": ["stomach cancer", "gastric adenocarcinoma", "gastric carcinoma"],
    "hepatocellular carcinoma": ["HCC", "liver cancer"],
    "renal cell carcinoma": ["RCC", "kidney cancer", "clear cell renal cell carcinoma", "ccRCC"],
    "bladder cancer": ["urothelial carcinoma", "urothelial cancer"],
    "glioblastoma": ["GBM", "glioblastoma multiforme"],
    "glioma": ["low-grade glioma", "diffuse glioma"],
    "acute myeloid leukemia": ["AML", "acute myeloid leukaemia"],
    "chronic myeloid leukemia": ["CML", "chronic myeloid leukaemia"],
    "chronic lymphocytic leukemia": ["CLL", "chronic lymphocytic leukaemia"],
    "acute lymphoblastic leukemia": ["acute lymphoblastic leukaemia"],
    "multiple myeloma": [],
    "diffuse large B-cell lymphoma": ["DLBCL"],
    "lymphoma": [],
    "head and neck squamous cell carcinoma": ["HNSCC"],
    "endometrial cancer": ["endometrial carcinoma", "uterine corpus endometrial carcinoma"],
    "cervical cancer": ["cervical carcinoma"],
    "esophageal cancer": ["oesophageal cancer", "esophageal carcinoma", "esophageal adenocarcinoma"],
    "thyroid cancer": ["thyroid carcinoma", "papillary thyroid carcinoma"],
    "cholangiocarcinoma": ["biliary tract cancer", "bile duct cancer"],
    "neuroblastoma": [],
    "sarcoma": ["osteosarcoma", "soft tissue sarcoma"],
    "mesothelioma": [],
    "gastrointestinal stromal tumor": ["GIST"],
}

# A specific subtype implies its parent for filtering purposes.
CANCER_PARENTS: dict[str, str] = {
    "triple-negative breast cancer": "breast cancer",
    "non-small cell lung cancer": "lung cancer",
    "small cell lung cancer": "lung cancer",
    "glioblastoma": "glioma",
}

_VARIANT_RE = re.compile(r"\b(?:p\.)?([ACDEFGHIKLMNPQRSTVWY])(\d{2,4})([ACDEFGHIKLMNPQRSTVWY*])\b")


def _alternation(terms: list[str]) -> str:
    return "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))


def _is_acronym(term: str) -> bool:
    letters = [c for c in term if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(term) <= 8


@dataclass(frozen=True)
class Entities:
    genes: list[str] = field(default_factory=list)
    cancer_types: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)

    def as_set(self) -> set[str]:
        return {*self.genes, *self.cancer_types}


class EntityExtractor:
    def __init__(self) -> None:
        self._gene_lookup: dict[str, str] = {}
        for canonical, aliases in GENES.items():
            for term in [canonical, *aliases]:
                self._gene_lookup[term] = canonical
        self._gene_re = re.compile(
            rf"(?<![A-Za-z0-9])(?:{_alternation(list(self._gene_lookup))})(?![A-Za-z0-9])"
        )

        self._cancer_lookup: dict[str, str] = {}
        acronyms: list[str] = []
        phrases: list[str] = []
        for canonical, aliases in CANCER_TYPES.items():
            for term in [canonical, *aliases]:
                if _is_acronym(term):
                    self._cancer_lookup[term] = canonical
                    acronyms.append(term)
                else:
                    self._cancer_lookup[term.lower()] = canonical
                    phrases.append(term)
        self._cancer_acronym_re = re.compile(
            rf"(?<![A-Za-z0-9])(?:{_alternation(acronyms)})(?![A-Za-z0-9])"
        )
        self._cancer_phrase_re = re.compile(
            rf"(?<![A-Za-z0-9])(?:{_alternation(phrases)})(?![A-Za-z0-9])", re.IGNORECASE
        )

    # -- public API ----------------------------------------------------
    def extract(self, text: str) -> Entities:
        genes = sorted({self._gene_lookup[m.group(0)] for m in self._gene_re.finditer(text)})

        cancers: set[str] = set()
        for m in self._cancer_acronym_re.finditer(text):
            cancers.add(self._cancer_lookup[m.group(0)])
        for m in self._cancer_phrase_re.finditer(text):
            cancers.add(self._cancer_lookup[m.group(0).lower()])
        for child in list(cancers):
            if child in CANCER_PARENTS:
                cancers.add(CANCER_PARENTS[child])

        # Variants are only trusted next to a gene mention; bare "T790M"-shaped tokens elsewhere
        # (cell-line names, reagent codes) are noise.
        variants: list[str] = []
        if genes:
            variants = sorted({"".join(m.groups()) for m in _VARIANT_RE.finditer(text)})
        return Entities(genes=genes, cancer_types=sorted(cancers), variants=variants)

    def normalize_gene(self, name: str) -> str | None:
        return self._gene_lookup.get(name.strip()) or self._gene_lookup.get(name.strip().upper())

    def normalize_cancer(self, name: str) -> str | None:
        key = name.strip()
        return (
            self._cancer_lookup.get(key)
            or self._cancer_lookup.get(key.lower())
            or (key.lower() if key.lower() in CANCER_PARENTS.values() else None)
        )


_default: EntityExtractor | None = None


def get_extractor() -> EntityExtractor:
    global _default
    if _default is None:
        _default = EntityExtractor()
    return _default
