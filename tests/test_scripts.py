"""Tests for scripts that edit local files. No real .env is touched."""

import base64
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "generate_airflow_keys", Path(__file__).resolve().parent.parent / "scripts" / "generate_airflow_keys.py"
)
keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keys)

PLACEHOLDERS = (
    "OTHER=1\n"
    "AIRFLOW_SECRET_KEY=generate_a_random_hex_string_with_python_secrets_module\n"
    "AIRFLOW_FERNET_KEY=generate_a_fernet_key_with_cryptography_library\n"
)


def _value(text, name):
    return next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith(name + "="))


def test_placeholders_are_replaced_with_valid_keys():
    text, written, kept = keys.fill_keys(PLACEHOLDERS)
    assert written == ["AIRFLOW_SECRET_KEY", "AIRFLOW_FERNET_KEY"] and kept == []
    secret, fernet = _value(text, "AIRFLOW_SECRET_KEY"), _value(text, "AIRFLOW_FERNET_KEY")
    assert len(secret) == 64 and int(secret, 16) >= 0
    assert len(base64.urlsafe_b64decode(fernet)) == 32
    assert text.startswith("OTHER=1\n")


def test_existing_keys_are_kept_unless_forced():
    first, _, _ = keys.fill_keys(PLACEHOLDERS)
    again, written, kept = keys.fill_keys(first)
    assert again == first and written == [] and len(kept) == 2

    forced, written, _ = keys.fill_keys(first, force=True)
    assert len(written) == 2 and _value(forced, "AIRFLOW_FERNET_KEY") != _value(first, "AIRFLOW_FERNET_KEY")


def test_missing_lines_are_added():
    text, written, _ = keys.fill_keys("OTHER=1\n")
    assert written == ["AIRFLOW_SECRET_KEY", "AIRFLOW_FERNET_KEY"]
    assert "AIRFLOW_FERNET_KEY=" in text
