"""TOON v4.1 JSON-model encoder, comma delimiter, two-space indentation.

No decoder or non-JSON host coercion is exposed. Integer precision is arbitrary;
floats use Python's shortest round-trippable decimal representation.
"""

import math
import re
from decimal import Decimal
from typing import Any


def quoted(value: str) -> str:
    value.encode("utf-8", errors="strict")
    escapes = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    return (
        '"' + "".join(escapes.get(c, f"\\u{ord(c):04x}" if ord(c) < 32 else c) for c in value) + '"'
    )


def key(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", value) else quoted(value)


def scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "null"
        if value == 0:
            return "0"
        if 1e-6 <= abs(value) < 1e21:
            number = format(Decimal(str(value)), "f")
            return number.rstrip("0").rstrip(".") if "." in number else number
        return str(value)
    if not isinstance(value, str):
        raise TypeError("TOON supports JSON values only")
    value.encode("utf-8", errors="strict")
    if (
        not value
        or value.strip() != value
        or value in {"true", "false", "null"}
        or value.startswith(("-", "#"))
        or re.search(r'[:,"\\\[\]{}\x00-\x1f]', value)
        or re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?", value, re.I)
    ):
        return quoted(value)
    return value


def primitive(value: Any) -> bool:
    return not isinstance(value, dict | list)


def columns(rows: list[Any]) -> list[tuple[str, Any]] | None:
    if not rows or any(not isinstance(r, dict) or not r for r in rows):
        return None
    names = list(rows[0])
    if any(set(r) != set(names) for r in rows):
        return None
    out: list[tuple[str, Any]] = []
    for name in names:
        values = [r[name] for r in rows]
        if all(primitive(v) for v in values):
            out.append((name, None))
        else:
            nested = columns(values)
            if nested is None:
                return None
            out.append((name, nested))
    return out


def header(cols: list[tuple[str, Any]]) -> str:
    return ",".join(key(k) + ("{" + header(sub) + "}" if sub else "") for k, sub in cols)


def cells(row: dict[str, Any], cols: list[tuple[str, Any]]) -> list[str]:
    result = []
    for name, sub in cols:
        result.extend(cells(row[name], sub) if sub else [scalar(row[name])])
    return result


def encode(value: Any) -> str:
    def render(
        node: Any,
        depth: int,
        name: str = "",
        *,
        list_item: bool = False,
        anonymous_object: bool = False,
    ) -> list[str]:
        indent = "  " * depth
        prefix = key(name) if name else ""
        if isinstance(node, dict):
            cols = columns(list(node.values())) if len(node) >= 2 and not anonymous_object else None
            if cols:
                lines = [indent + prefix + f"[{len(node)}:]{{{header(cols)}}}:"]
                return lines + [
                    indent + "  " + key(k) + ": " + ",".join(cells(v, cols))
                    for k, v in node.items()
                ]
            lines = [indent + prefix + ":"] if name else []
            for k, v in node.items():
                lines.extend(render(v, depth + (1 if name else 0), k))
            return lines
        if isinstance(node, list):
            if not node:
                return [indent + (prefix + ": []" if name else "[0]:" if list_item else "[]")]
            cols = columns(node) if not list_item else None
            head = indent + prefix + f"[{len(node)}]"
            if cols:
                return [head + "{" + header(cols) + "}:"] + [
                    indent + "  " + ",".join(cells(row, cols)) for row in node
                ]
            if all(primitive(v) for v in node):
                return [head + ": " + ",".join(scalar(v) for v in node)]
            lines = [head + ":"]
            for item in node:
                if isinstance(item, dict):
                    child = render(item, depth + 2, anonymous_object=True)
                    if not child:
                        lines.append(indent + "  -")
                    else:
                        lines.append(indent + "  - " + child[0].lstrip())
                        lines.extend(child[1:])
                else:
                    child = render(item, depth + 1, list_item=True)
                    lines.append(indent + "  - " + child[0].lstrip())
                    lines.extend(child[1:])
            return lines
        return [indent + (prefix + ": " if name else "") + scalar(node)]

    return "\n".join(render(value, 0))
