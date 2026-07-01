from __future__ import annotations

import os
import unittest
import base64
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from scripts.dashboard_app import _basic_credentials_from_header, require_dashboard_access


def _request(ip: str = "127.0.0.1") -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host=ip))


class DashboardAuthTest(unittest.TestCase):
    def test_basic_credentials_are_parsed_from_header(self) -> None:
        token = base64.b64encode(b"admin:secret-dashboard-password").decode("ascii")

        credentials = _basic_credentials_from_header(f"Basic {token}")

        self.assertEqual(credentials, ("admin", "secret-dashboard-password"))

    def test_invalid_basic_credentials_header_is_rejected(self) -> None:
        self.assertIsNone(_basic_credentials_from_header(None))
        self.assertIsNone(_basic_credentials_from_header("Bearer token"))
        self.assertIsNone(_basic_credentials_from_header("Basic not-base64"))

    def test_dashboard_requires_configured_password(self) -> None:
        with patch.dict(os.environ, {"DASHBOARD_USERNAME": "admin"}, clear=False):
            os.environ.pop("DASHBOARD_PASSWORD", None)
            credentials = HTTPBasicCredentials(username="admin", password="anything")

            with self.assertRaises(HTTPException) as ctx:
                require_dashboard_access(_request(), credentials)

            self.assertEqual(ctx.exception.status_code, 503)

    def test_dashboard_accepts_valid_basic_auth(self) -> None:
        with patch.dict(
            os.environ,
            {"DASHBOARD_USERNAME": "admin", "DASHBOARD_PASSWORD": "secret-dashboard-password"},
            clear=False,
        ):
            credentials = HTTPBasicCredentials(username="admin", password="secret-dashboard-password")

            user = require_dashboard_access(_request(), credentials)

            self.assertEqual(user, "admin")

    def test_dashboard_rejects_disallowed_ip(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHBOARD_USERNAME": "admin",
                "DASHBOARD_PASSWORD": "secret-dashboard-password",
                "DASHBOARD_ALLOWED_IPS": "203.0.113.10",
            },
            clear=False,
        ):
            credentials = HTTPBasicCredentials(username="admin", password="secret-dashboard-password")

            with self.assertRaises(HTTPException) as ctx:
                require_dashboard_access(_request("127.0.0.1"), credentials)

            self.assertEqual(ctx.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
