from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

from scripts.common import CONFIG_DIR, REPORTS_DIR, ensure_dirs

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrivacyPolicy:
    enabled: bool
    allowed_domains: tuple[str, ...]
    audit_log: Path


def load_policy(path: Path | None = None) -> PrivacyPolicy:
    policy_path = path or CONFIG_DIR / "privacy_policy.yaml"
    with policy_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    audit_log = REPORTS_DIR / Path(data.get("audit_log", "data/reports/outbound_audit.log")).name
    return PrivacyPolicy(
        enabled=bool(data.get("privacy_mode", True)),
        allowed_domains=tuple(data.get("allowed_domains", [])),
        audit_log=audit_log,
    )


def _domain_allowed(domain: str, allowed_domains: tuple[str, ...]) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains)


def audit_request(method: str, url: str, allowed: bool, reason: str, policy: PrivacyPolicy | None = None) -> None:
    ensure_dirs()
    policy = policy or load_policy()
    parsed = urlparse(url)
    policy.audit_log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not policy.audit_log.exists()
    with policy.audit_log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "method", "domain", "path", "allowed", "reason"])
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": method.upper(),
                "domain": parsed.netloc.lower(),
                "path": parsed.path,
                "allowed": str(allowed).lower(),
                "reason": reason,
            }
        )


def checked_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    policy = load_policy()
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    allowed = (not policy.enabled) or _domain_allowed(domain, policy.allowed_domains)
    reason = "allowlisted" if allowed else "blocked_by_privacy_allowlist"
    audit_request(method, url, allowed=allowed, reason=reason, policy=policy)
    if not allowed:
        raise RuntimeError(f"Outbound request blocked by privacy policy: {domain}")
    timeout = kwargs.pop("timeout", 10)
    return requests.request(method=method, url=url, timeout=timeout, **kwargs)


def install_requests_monkeypatch() -> None:
    original_request = requests.sessions.Session.request

    def audited_request(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        policy = load_policy()
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        allowed = (not policy.enabled) or _domain_allowed(domain, policy.allowed_domains)
        reason = "allowlisted" if allowed else "blocked_by_privacy_allowlist"
        audit_request(method, url, allowed=allowed, reason=reason, policy=policy)
        if not allowed:
            raise RuntimeError(f"Outbound request blocked by privacy policy: {domain}")
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = audited_request
    LOGGER.info("Installed requests outbound privacy audit monkeypatch.")


def privacy_check() -> None:
    policy = load_policy()
    ensure_dirs()
    LOGGER.info("Privacy mode=%s allowed_domains=%s audit_log=%s", policy.enabled, policy.allowed_domains, policy.audit_log)
    audit_request("GET", "https://example.com/privacy-check", allowed=False, reason="self_test_blocked_domain", policy=policy)


if __name__ == "__main__":
    from scripts.common import setup_logging

    setup_logging()
    privacy_check()

