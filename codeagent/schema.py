import json
from collections.abc import Collection
from typing import Any

_PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def decode_arguments(raw: str) -> Any:
    """Parse the JSON arguments of a tool call; an empty string means no arguments. Raises ValueError."""
    return json.loads(raw.strip() or "{}")


def object_schema(properties: dict[str, Any], required: Collection[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def function_spec(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Check a decoded JSON value against the JSON Schema subset used by tool parameters and return every violation."""
    expected = schema.get("type")
    if expected is not None and not _has_type(value, expected):
        return [f"{path} must be of type {expected}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of: {', '.join(map(str, schema['enum']))}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path} must have at least {schema['minLength']} characters")
    if isinstance(value, list):
        errors.extend(_validate_array(value, schema, path))
    if isinstance(value, dict):
        errors.extend(_validate_object(value, schema, path))
    return errors


def _has_type(value: Any, expected: str) -> bool:
    if isinstance(value, bool) and expected in ("integer", "number"):
        return False
    return isinstance(value, _PYTHON_TYPES[expected])


def _validate_array(value: list[Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if len(value) < schema.get("minItems", 0):
        errors.append(f"{path} must have at least {schema['minItems']} items")
    items = schema.get("items")
    if items is not None:
        for index, item in enumerate(value):
            errors.extend(validate(item, items, f"{path}[{index}]"))
    return errors


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
    properties = schema.get("properties", {})
    errors = [f"{path}.{key} is required" for key in schema.get("required", ()) if key not in value]
    for key, item in value.items():
        if key in properties:
            errors.extend(validate(item, properties[key], f"{path}.{key}"))
        elif schema.get("additionalProperties") is False:
            errors.append(f"{path}.{key} is not allowed")
    return errors
