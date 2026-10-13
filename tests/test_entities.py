from oncorag.ingestion.entities import get_extractor

ex = get_extractor()


def test_alias_normalisation():
    ents = ex.extract("p53 and HER2 status were assessed; PD-L1 was high.")
    assert ents.genes == ["CD274", "ERBB2", "TP53"]


def test_gene_symbols_are_case_sensitive():
    assert ex.extract("The patients met criteria after a kit was used").genes == []
    assert ex.extract("MET amplification drives resistance").genes == ["MET"]


def test_longest_match_prevents_double_counting_nsclc():
    ents = ex.extract("Patients with non-small cell lung cancer were enrolled.")
    assert "non-small cell lung cancer" in ents.cancer_types
    assert "small cell lung cancer" not in ents.cancer_types


def test_parent_cancer_types_are_added():
    ents = ex.extract("TNBC is aggressive.")
    assert {"triple-negative breast cancer", "breast cancer"} <= set(ents.cancer_types)


def test_hyphenated_context_still_matches():
    assert "KRAS" in ex.extract("KRAS-mutant tumors and pan-KRAS inhibitors").genes
    assert "ALK" in ex.extract("EML4-ALK fusions").genes


def test_variants_only_trusted_next_to_a_gene():
    assert ex.extract("EGFR T790M and L858R mutations").variants == ["L858R", "T790M"]
    assert ex.extract("The reagent code was B123X in this batch").variants == []


def test_acronym_cancer_matching_is_case_sensitive():
    assert "colorectal cancer" in ex.extract("CRC outcomes").cancer_types
    assert ex.extract("the crc of the matter").cancer_types == []


def test_normalize_helpers():
    assert ex.normalize_gene("p53") == "TP53"
    assert ex.normalize_gene("kras") == "KRAS"
    assert ex.normalize_cancer("NSCLC") == "non-small cell lung cancer"
    assert ex.normalize_cancer("lung cancer") == "lung cancer"
    assert ex.normalize_gene("not-a-gene") is None
