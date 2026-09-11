"""O1: Copernicus Data Space Ecosystem (CDSE) OAuth2 authentication.

CDSE uses Keycloak OAuth2 password-grant for personal accounts. Set
CDSE_USERNAME and CDSE_PASSWORD as environment variables (or a .env file --
never commit credentials to the repo).
"""

from __future__ import annotations

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


class CDSESession:
    def __init__(self, username: str | None = None, password: str | None = None):
        self.username = username or os.environ.get("CDSE_USERNAME")
        self.password = password or os.environ.get("CDSE_PASSWORD")
        if not self.username or not self.password:
            raise ValueError(
                "CDSE credentials not found. Set CDSE_USERNAME / CDSE_PASSWORD "
                "env vars after registering at https://dataspace.copernicus.eu"
            )
        self._access_token = None
        self._refresh_token = None
        self._expires_at = 0

    def _request_token(self):
        resp = requests.post(
            TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "username": self.username,
                "password": self.password,
                "grant_type": "password",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload["access_token"]
        self._refresh_token = payload["refresh_token"]
        self._expires_at = time.time() + payload["expires_in"] - 30

    def get_token(self) -> str:
        if not self._access_token or time.time() >= self._expires_at:
            self._request_token()
        return self._access_token

    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}
