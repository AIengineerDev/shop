from pathlib import Path

import pytest

from catalog_onboard.config import DEFAULT_RULES_DIR
from catalog_onboard.rules import RULE_FILES, compute_rules_version, load_rules


def test_rules_version_changes_with_content(tmp_path: Path):
    for name in RULE_FILES:
        (tmp_path / name).write_bytes((DEFAULT_RULES_DIR / name).read_bytes())
    v1 = compute_rules_version(tmp_path)
    assert v1 == compute_rules_version(DEFAULT_RULES_DIR)
    (tmp_path / "global_prohibited.yaml").write_text("keywords: [foo]\n")
    assert compute_rules_version(tmp_path) != v1


def test_missing_file_is_config_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_rules(tmp_path)


def test_tag_mapping_scopes(rules):
    assert rules.tag_key("sup_acme", "CLR") == "color"
    assert rules.tag_key("sup_other", "clr") is None
    assert rules.tag_key("sup_other", "colour") == "color"
    assert rules.tag_value("sup_acme", "size", "lg") == "L"
    assert rules.tag_value("sup_other", "color", "NVY") == "navy"
    assert rules.tag_value("sup_other", "color", "teal") == "teal"
