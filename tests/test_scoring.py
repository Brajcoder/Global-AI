from packages.scoring_engine.scoring import chronos_score, eff_score


def test_eff_score_bounds():
    val = eff_score(tokens_used=2000, latency_ms=5000, loop_ratio=0.1)
    assert 0 <= val <= 100


def test_chronos_score_weighting():
    val = chronos_score(truth=80, efficiency=60)
    assert round(val, 2) == 73.0


def test_missing_publish_date_is_neutral_not_perfect():
    from datetime import datetime, timezone

    from packages.scoring_engine.scoring import UNKNOWN_DATE_FRESHNESS, freshness_score

    assert freshness_score(None, datetime.now(timezone.utc), 90) == UNKNOWN_DATE_FRESHNESS
