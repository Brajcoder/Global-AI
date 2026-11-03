from packages.scoring_engine.scoring import chronos_score, eff_score


def test_eff_score_bounds():
    val = eff_score(tokens_used=2000, latency_ms=5000, loop_ratio=0.1)
    assert 0 <= val <= 100


def test_chronos_score_weighting():
    val = chronos_score(truth=80, efficiency=60)
    assert round(val, 2) == 73.0
