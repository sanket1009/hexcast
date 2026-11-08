"""
generate_airflow_keys.py
========================
Fill in the two Airflow keys in .env with fresh random values.

    AIRFLOW_SECRET_KEY  signs the tokens Airflow's API server, scheduler and
                        workers use to trust each other, and UI login sessions
    AIRFLOW_FERNET_KEY  encrypts passwords Airflow stores in its database
                        (connections such as the SMTP login)

Only placeholder or empty values are replaced, so running it again never
changes a key that is already set. A changed Fernet key would leave Airflow
unable to read what it encrypted before; use --force only on a fresh install.
The keys are written to .env and never printed.

Usage (from the project folder):
    python3 scripts/generate_airflow_keys.py
    python3 scripts/generate_airflow_keys.py --force   # replace existing keys too
"""

import argparse
import base64
import re
import secrets
import sys
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PLACEHOLDER = re.compile(r"^(|generate_.*|change_?me.*)$", re.IGNORECASE)


def new_secret_key() -> str:
    """64 hex characters (32 random bytes)."""
    return secrets.token_hex(32)


def new_fernet_key() -> str:
    """32 random bytes, URL-safe base64: the format Fernet requires."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


GENERATORS = {
    "AIRFLOW_SECRET_KEY": new_secret_key,
    "AIRFLOW_FERNET_KEY": new_fernet_key,
}


def fill_keys(text: str, force: bool = False) -> tuple[str, list[str], list[str]]:
    """Return the new .env text, the keys written, and the keys left as they were."""
    written, kept = [], []
    for name, make in GENERATORS.items():
        line = re.compile(rf"^{name}=(.*)$", re.MULTILINE)
        match = line.search(text)
        if match is None:
            text = text.rstrip("\n") + f"\n{name}={make()}\n"
            written.append(name)
        elif force or PLACEHOLDER.match(match.group(1).strip()):
            text = line.sub(f"{name}={make()}", text, count=1)
            written.append(name)
        else:
            kept.append(name)
    return text, written, kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace keys that are already set")
    args = parser.parse_args()

    if not ENV_PATH.exists():
        print(f"No .env at {ENV_PATH}. Copy .env.example to .env first.")
        return 1

    text, written, kept = fill_keys(ENV_PATH.read_text(), force=args.force)
    ENV_PATH.write_text(text)
    for name in written:
        print(f"Set {name}")
    for name in kept:
        print(f"Kept {name} (already set; use --force to replace)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
