from packages.evidence_engine.stancer import infer_stance, outdated_terms_in_claim, source_quality_score

MIGRATION = "Pydantic V2 replaces dict() with model_dump() and validator with field_validator."


def test_outdated_usage_contradicts():
    assert infer_stance("Pydantic serialization uses .dict()", MIGRATION) == "contradicts"
    terms = {t.old: t.new for t in outdated_terms_in_claim("use .dict() and @validator", MIGRATION)}
    assert terms == {"dict": "model_dump", "validator": "field_validator"}


def test_modern_usage_is_not_contradicted():
    assert infer_stance("Pydantic V2 uses model_dump() for serialization", MIGRATION) != "contradicts"


def test_substring_is_does_not_imply_support():
    # Regression: "is" was a support hint and matched "this"/"lists".
    assert infer_stance("The CEO of Acme is Alice Smith", "This page lists the board members") == "irrelevant"


def test_overlap_supports():
    assert infer_stance("FastAPI supports async endpoints", "FastAPI supports async endpoints natively.") == "supports"


def test_source_quality_matches_host_suffix_not_substring():
    assert source_quality_score("https://docs.pydantic.dev/latest/") == 100.0
    assert source_quality_score("https://pypi.org/project/x") == 100.0
    assert source_quality_score("https://notpypi.org.evil.com/") == 35.0
    assert source_quality_score("https://github.com/x/y") == 70.0
