from codeagent.schema import validate

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string", "minLength": 1}, "count": {"type": "integer"}},
    "required": ["path"],
    "additionalProperties": False,
}


def test_valid_arguments_pass():
    assert validate({"path": "order.py", "count": 2}, SCHEMA) == []


def test_every_violation_is_reported():
    errors = validate({"count": True, "extra": 1}, SCHEMA)
    assert "$.path is required" in errors
    assert "$.count must be of type integer" in errors
    assert "$.extra is not allowed" in errors


def test_nested_arrays_are_checked():
    schema = {"type": "array", "minItems": 1, "items": {"type": "string", "enum": ["a", "b"]}}
    assert validate([], schema) == ["$ must have at least 1 items"]
    assert validate(["a", "c"], schema) == ["$[1] must be one of: a, b"]
