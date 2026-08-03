import pytest
from toponymy import new_templates

def _template_module():
    return new_templates

def test_prompt_schema_is_owned_and_inspection_is_defensive():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    prompt = _template_module().Prompt("s", "u", schema)
    schema["properties"]["name"]["type"] = "integer"
    exposed = prompt.json_schema
    exposed["properties"]["name"]["type"] = "array"
    assert prompt.json_schema["properties"]["name"]["type"] == "string"
    prompt._asdict()["json_schema"].clear()
    assert prompt.json_schema["type"] == "object"
    with pytest.raises(AttributeError):
        prompt.user = "changed"
