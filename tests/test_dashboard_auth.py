from __future__ import annotations

import os
import unittest
import math
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from scripts.dashboard_app import _check_dashboard_access, _create_session_token, _session_username, _with_kyiv_times


def _request(ip: str = "127.0.0.1") -> SimpleNamespace:
    return SimpleNamespace(client=SimpleNamespace(host=ip))


class DashboardAuthTest(unittest.TestCase):
    def test_dashboard_payload_converts_non_finite_numbers_to_none(self) -> None:
        payload = _with_kyiv_times({"profit_factor": float("inf"), "bad": float("nan"), "balance": 1000.0})

        self.assertIsNone(payload["profit_factor"])
        self.assertIsNone(payload["bad"])
        self.assertEqual(payload["balance"], 1000.0)
        self.assertFalse(any(isinstance(value, float) and math.isnan(value) for value in payload.values()))

    def test_session_token_roundtrip(self) -> None:
        with patch.dict(
            os.environ,
            {"DASHBOARD_PASSWORD": "secret-dashboard-password", "DASHBOARD_SESSION_SECONDS": "86400"},
            clear=False,
        ):
            token = _create_session_token("admin", issued_at=1000)

            self.assertEqual(_session_username(token, now=1001), "admin")

    def test_expired_session_token_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"DASHBOARD_PASSWORD": "secret-dashboard-password", "DASHBOARD_SESSION_SECONDS": "10"},
            clear=False,
        ):
            token = _create_session_token("admin", issued_at=1000)

            self.assertIsNone(_session_username(token, now=1011))

    def test_dashboard_requires_configured_password(self) -> None:
        with patch.dict(os.environ, {"DASHBOARD_USERNAME": "admin"}, clear=False):
            os.environ.pop("DASHBOARD_PASSWORD", None)

            with self.assertRaises(HTTPException) as ctx:
                _check_dashboard_access(_request(), "admin", "anything")

            self.assertEqual(ctx.exception.status_code, 503)

    def test_dashboard_accepts_valid_basic_auth(self) -> None:
        with patch.dict(
            os.environ,
            {"DASHBOARD_USERNAME": "admin", "DASHBOARD_PASSWORD": "secret-dashboard-password"},
            clear=False,
        ):
            user = _check_dashboard_access(_request(), "admin", "secret-dashboard-password")

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
            with self.assertRaises(HTTPException) as ctx:
                _check_dashboard_access(_request("127.0.0.1"), "admin", "secret-dashboard-password")

            self.assertEqual(ctx.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
