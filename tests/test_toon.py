from reflex_axi.toon import scalar


# Expected values are the official TOON spec fixtures
# (toon-format/spec tests/fixtures/encode/primitives.json): "encodes small decimal without
# exponent notation", "encodes large number" and "encodes zero" (negative zero).
def test_canonical_numbers():
    assert scalar(1e-6) == "0.000001"
    assert scalar(1e20) == "100000000000000000000"
    assert scalar(-0.0) == "0"
