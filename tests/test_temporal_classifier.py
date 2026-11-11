from packages.temporal_engine.classifier import classify_claim


def test_time_sensitive_detection():
    d = classify_claim("Latest API version for package x is 2.0")
    assert d.time_sensitive is True
    assert d.verify_live is True


def test_stable_detection():
    d = classify_claim("Binary search runs in O(log n).")
    assert d.verify_live is False


def test_code_api_usage_is_checked_live():
    assert classify_claim("To serialize in Pydantic, use .dict()").verify_live is True
    assert classify_claim("Decorate it with @validator").verify_live is True


def test_word_boundaries():
    # Regression: "deliver" matched "live", "rapid" matched "api", "conversion" matched "version".
    assert classify_claim("Teams deliver rapid unit conversion tools.").verify_live is False


def test_hedged_checkable_claim_is_still_checked():
    assert classify_claim("GPT-4o pricing is likely the best value").verify_live is True
