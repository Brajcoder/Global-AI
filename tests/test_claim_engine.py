from packages.claim_engine.extractor import extract_atomic_claims


def test_conjunctions_do_not_fragment_claims():
    claims = extract_atomic_claims("To serialize in Pydantic, use .dict() and @validator.")
    assert [c.text for c in claims] == ["To serialize in Pydantic, use .dict() and @validator."]


def test_offsets_are_exact():
    text = "First claim is here today. Second claim is also here; third clause is right here."
    claims = extract_atomic_claims(text)
    assert len(claims) == 3
    for c in claims:
        assert text[c.start:c.end] == c.text


def test_abbreviations_and_versions_do_not_split():
    text = "Use a framework, e.g. FastAPI version 0.116 or later."
    assert len(extract_atomic_claims(text)) == 1


def test_max_claims_respected():
    text = " ".join(f"Sentence number {i} is a claim." for i in range(20))
    assert len(extract_atomic_claims(text, max_claims=4)) == 4
