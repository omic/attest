"""Fernet-encrypted token persistence alongside the .attest database."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None


class TokenStore:
    """Stores OAuth tokens in a Fernet-encrypted JSON file at {db_path}.tokens."""

    def __init__(self, db_path: str):
        if Fernet is None:
            raise ImportError(
                "pip install cryptography for encrypted token storage"
            )
        self._path = Path(db_path).with_suffix(".tokens")
        self._key_path = Path(db_path).with_suffix(".token_key")
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self._key_path.exists():
            return self._key_path.read_bytes()
        key = Fernet.generate_key()
        self._key_path.write_bytes(key)
        os.chmod(str(self._key_path), 0o600)
        return key

    def _load_all(self) -> dict:
        if not self._path.exists():
            return {}
        encrypted = self._path.read_bytes()
        return json.loads(self._fernet.decrypt(encrypted))

    def _save_all(self, data: dict) -> None:
        encrypted = self._fernet.encrypt(json.dumps(data).encode())
        self._path.write_bytes(encrypted)
        os.chmod(str(self._path), 0o600)

    def save_token(self, provider: str, data: dict) -> None:
        all_tokens = self._load_all()
        data["saved_at"] = int(time.time())
        all_tokens[provider] = data
        self._save_all(all_tokens)

    def get_token(self, provider: str) -> dict | None:
        return self._load_all().get(provider)

    def delete_token(self, provider: str) -> bool:
        all_tokens = self._load_all()
        if provider in all_tokens:
            del all_tokens[provider]
            self._save_all(all_tokens)
            return True
        return False

    def list_providers(self) -> list[str]:
        return list(self._load_all().keys())

    def needs_refresh(self, provider: str) -> bool:
        token = self.get_token(provider)
        if not token or "expires_at" not in token:
            return False
        return token["expires_at"] < time.time() + 300
