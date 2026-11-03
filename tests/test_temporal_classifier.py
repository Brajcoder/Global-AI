from packages.temporal_engine.classifier import classify_claim


def test_time_sensitive_detection():
    d = classify_claim("Latest API version for package x is 2.0")
    assert d.time_sensitive is True
    assert d.verify_live is True


def test_stable_detection():
    d = classify_claim("Binary search runs in O(log n).")
    assert d.verify_live is False
