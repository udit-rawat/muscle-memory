import pytest

from mm.values import SecretStore, TemplateError, parameterize, referenced, render

SECRETS = SecretStore({"PW": "hunter22"})


def test_render_inputs_and_secrets() -> None:
    assert render("{{member_id}}/{{secret:PW}}", {"member_id": "10234"}, SECRETS) == "10234/hunter22"


def test_render_rejects_unknown_names() -> None:
    with pytest.raises(TemplateError):
        render("{{nope}}", {}, SECRETS)
    with pytest.raises(TemplateError):
        render("{{secret:NOPE}}", {}, SECRETS)


def test_parameterize_replaces_literal_input_values_longest_first() -> None:
    assert parameterize("go to 102345 then 10234", {"a": "10234", "b": "102345"}) == "go to {{b}} then {{a}}"


def test_referenced_splits_inputs_and_secrets() -> None:
    assert referenced("{{a}} {{secret:PW}} {{ b }}") == ({"a", "b"}, {"PW"})
