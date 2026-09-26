import pytest

from reflex_axi.identity import read_json
from reflex_axi.toon import encode, scalar


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"a": []}, "a: []"),
        ({"a": {}}, "a:"),
        ({"x": "true"}, 'x: "true"'),
        ({"x": "#comment"}, 'x: "#comment"'),
        ({"x": "001"}, 'x: "001"'),
        ({"x": "line\nnew"}, 'x: "line\\nnew"'),
        ({"x": "\b\f"}, 'x: "\\u0008\\u000c"'),
        ({"x": "\\b"}, 'x: "\\\\b"'),
        ({"a-b": "ok"}, '"a-b": ok'),
        ([{"id": 1, "x": "a"}, {"id": 2, "x": "b"}], "[2]{id,x}:\n  1,a\n  2,b"),
        ({"one": {"x": 1}, "two": {"x": 2}}, "[2:]{x}:\n  one: 1\n  two: 2"),
        ([{"x": {"y": 1}}, {"x": {"y": 2}}], "[2]{x{y}}:\n  1\n  2"),
        ([{"x": [1, 2], "y": "yes"}, {}], "[2]:\n  - x[2]: 1,2\n    y: yes\n  -"),
        ([[1], []], "[2]:\n  - [1]: 1\n  - [0]:"),
    ],
)
def test_toon_vectors(value, expected):
    assert encode(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1e-6, "0.000001"),
        (1e20, "100000000000000000000"),
        (1.0, "1"),
        (-0.0, "0"),
        (0.00120, "0.0012"),
    ],
)
def test_canonical_numbers(value, expected):
    assert scalar(value) == expected


def test_surrogates_rejected():
    with pytest.raises(UnicodeEncodeError):
        encode({"x": "\ud800"})


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        read_json(text)
