"""把客户端（OpenAI 风格）工具定义里的 JSON Schema 收敛成 Gemini 接受的形状。

背景：google-genai 的 ``types.Schema`` 只保留 Gemini OpenAPI 子集里的 25 个字段，
且 pydantic 配置是 ``extra="forbid"``。客户端（酒馆预设、MCP、各类插件）自动生成的
工具定义里常见 ``oneOf`` / ``$ref`` / ``$defs`` / ``additionalProperties`` / ``const``
等关键字，这些字段在 ``GenerateContentConfig`` 构造时就会被本地校验拒掉——
请求根本没发出去，流式响应于是直接中断。典型报错：

    tools.function_declarations.17.parameters.properties.at.oneOf
      Extra inputs are not permitted [type=extra_forbidden]

这里做一次**有损但安全**的降级，保证请求一定能出站：
1. ``$ref`` 就地展开（带 JSON Pointer 解析与环检测，depth/节点预算兜底）；
2. ``allOf`` 合并、``oneOf`` / ``anyOf`` 收敛成单一类型（对象分支合并属性，
   必填项取交集），``type`` 数组里的 ``null`` 转成 ``nullable``；
3. 丢弃 Gemini 不认识的字段，``enum`` 统一成字符串数组。
"""

import json
from typing import Any, Optional

# 递归与规模上限：schema_validation 已在上游卡过 32 层 / 10000 节点，
# 这里给转换结果更紧的预算，避免递归 schema 展开后把请求体撑爆。
MAX_DEPTH = 16
MAX_NODES = 2000

_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}
_STRING_KEYS = ("description", "title", "format", "pattern")
_NUMBER_KEYS = ("minimum", "maximum")
_INT_KEYS = ("minItems", "maxItems", "minLength", "maxLength")


def to_gemini_schema(schema: Any) -> dict:
    """返回只含 Gemini Schema 字段的参数定义；无法识别的输入回退成空 schema。"""
    if not isinstance(schema, dict):
        return {}
    budget = [MAX_NODES]
    converted = _convert(schema, schema, 0, budget)
    return converted if isinstance(converted, dict) else {}


def _convert(node: Any, root: dict, depth: int, budget: list) -> dict:
    if not isinstance(node, dict) or depth > MAX_DEPTH or budget[0] <= 0:
        return {}
    budget[0] -= 1

    node = _deref(node, root, ())
    if not isinstance(node, dict):
        return {}
    node = _flatten(node, root, depth, budget)

    out: dict = {}
    type_name, nullable = _normalise_type(node.get("type"))
    if type_name is None:
        if isinstance(node.get("properties"), dict) or node.get("additionalProperties") is not None:
            type_name = "object"
        elif "items" in node:
            type_name = "array"
    if type_name:
        out["type"] = type_name
    if nullable or node.get("nullable") is True:
        out["nullable"] = True

    for key in _STRING_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value

    enum_values = _enum_values(node)
    if enum_values:
        out["enum"] = enum_values

    for key in _NUMBER_KEYS:
        value = node.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    for key in _INT_KEYS:
        value = node.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[key] = value
    if "default" in node:
        out["default"] = node["default"]
    if "example" in node:
        out["example"] = node["example"]

    properties = node.get("properties")
    if isinstance(properties, dict) and properties:
        converted = {
            name: _convert(child, root, depth + 1, budget)
            for name, child in properties.items()
            if isinstance(name, str)
        }
        if converted:
            out["properties"] = converted
            required = node.get("required")
            if isinstance(required, list) and out.get("type") == "object":
                names = [name for name in _dedupe(required)
                         if isinstance(name, str) and name in converted]
                if names:
                    out["required"] = names

    items = node.get("items")
    if isinstance(items, list) and items:
        items = items[0]
    if isinstance(items, dict):
        converted_items = _convert(items, root, depth + 1, budget)
        if converted_items:
            out["items"] = converted_items

    return out


def _deref(node: dict, root: dict, visited: tuple) -> dict:
    """展开 ``$ref``；同级关键字按 JSON Schema 2020-12 的语义覆盖被引用节点。"""
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return node
    siblings = {key: value for key, value in node.items() if key != "$ref"}
    if ref in visited:
        return siblings
    target = _resolve_pointer(root, ref)
    if not isinstance(target, dict):
        return siblings
    merged = dict(target)
    merged.update(siblings)
    return _deref(merged, root, visited + (ref,))


def _resolve_pointer(root: dict, ref: str) -> Optional[dict]:
    if not ref.startswith("#"):
        return None
    pointer = ref[1:]
    if pointer in ("", "/"):
        return root
    node: Any = root
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    return node if isinstance(node, dict) else None


def _flatten(node: dict, root: dict, depth: int, budget: list) -> dict:
    """把 allOf / oneOf / anyOf 折叠掉，返回只含普通关键字的节点。"""
    node = dict(node)
    merged: dict = {}

    all_of = node.pop("allOf", None)
    if isinstance(all_of, list):
        for sub in all_of:
            if isinstance(sub, dict):
                merged = _merge(merged, _flatten(_deref(sub, root, ()), root, depth + 1, budget))

    for key in ("oneOf", "anyOf"):
        variants = node.pop(key, None)
        if isinstance(variants, list) and variants:
            merged = _merge(merged, _pick_variant(variants, root, depth, budget))

    return _merge(merged, node)


def _pick_variant(variants: list, root: dict, depth: int, budget: list) -> dict:
    """oneOf / anyOf 收敛：同构对象合并属性，其余取第一个非 null 分支。"""
    plain = [
        _flatten(item, root, depth + 1, budget)
        for item in variants
        if isinstance(item, dict)
    ]
    if not plain:
        return {}
    non_null = [item for item in plain if _normalise_type(item.get("type"))[0] != "null"]
    had_null = len(non_null) != len(plain)
    if not non_null:
        return {"nullable": True}

    if len(non_null) == 1:
        picked = dict(non_null[0])
        if had_null:
            picked["nullable"] = True
        return picked

    kinds = {_normalise_type(item.get("type"))[0] for item in non_null}
    if kinds == {"object"}:
        merged: dict = {}
        required_sets = []
        for item in non_null:
            merged = _merge(merged, item)
            required_sets.append({name for name in item.get("required") or [] if isinstance(name, str)})
        common = set.intersection(*required_sets) if required_sets else set()
        merged.pop("required", None)
        if common:
            merged["required"] = sorted(common)
        if had_null:
            merged["nullable"] = True
        return merged

    picked = dict(non_null[0])
    if had_null:
        picked["nullable"] = True
    return picked


def _merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if key == "properties" and isinstance(value, dict):
            existing = merged.get("properties")
            merged[key] = {**(existing if isinstance(existing, dict) else {}), **value}
        elif key == "required" and isinstance(value, list):
            merged[key] = _dedupe([*(merged.get(key) or []), *value])
        elif value is None and key in merged:
            continue
        else:
            merged[key] = value
    return merged


def _normalise_type(value: Any) -> tuple[Optional[str], bool]:
    """返回 (小写类型名, 是否可为空)；``["string", "null"]`` 这类联合类型取首个非空。"""
    if isinstance(value, str):
        name = value.strip().lower()
        if name == "null":
            return None, True
        return (name if name in _TYPES else None), False
    if isinstance(value, list):
        names = [item.strip().lower() for item in value if isinstance(item, str)]
        non_null = [name for name in names if name != "null"]
        nullable = len(non_null) < len(names)
        first = next((name for name in non_null if name in _TYPES), None)
        return first, nullable
    return None, False


def _enum_values(node: dict) -> list:
    raw = node.get("enum")
    if raw is None and "const" in node:
        raw = [node["const"]]
    if not isinstance(raw, list):
        return []
    values = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            text = item
        elif isinstance(item, (dict, list, bool)):
            text = json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        if text not in values:
            values.append(text)
    return values


def _dedupe(values: Any) -> list:
    seen = []
    for value in values or []:
        if value not in seen:
            seen.append(value)
    return seen
