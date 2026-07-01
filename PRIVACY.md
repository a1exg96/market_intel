# Privacy Model

Market Intelligence Trading Lab v0.1 is local-first.

## What Is Not Sent

The project does not send logs, configs, trades, model outputs, exceptions, API keys, strategies, system information, analytics, or telemetry to developers or third parties.

Telegram, Discord, Sentry, analytics, cloud telemetry, and remote logging are disabled by default.

## Allowed Outbound Domains

Outbound market-data requests are controlled by `configs/privacy_policy.yaml`. The default allowlist is:

- `binance.com`
- `api.binance.com`
- `fapi.binance.com`
- `bybit.com`
- `okx.com`
- `coinbase.com`
- `kraken.com`

Anything outside the allowlist is blocked by the project request wrapper.

## Audit Log

Every wrapped outbound request is recorded locally:

```text
data/reports/outbound_audit.log
```

Fields:

- `timestamp`
- `method`
- `domain`
- `path`
- `allowed`
- `reason`

Run this to create and inspect a privacy self-test entry:

```bash
python scripts/main.py privacy-check
type data\reports\outbound_audit.log
```

## API Keys

Do not put API keys in committed config files. Use `.env` or another local-only configuration file. The CLI refuses Freqtrade config files that include exchange keys or `dry_run=false`.

## External Integrations

News and on-chain APIs are off by default. If you enable them later, add explicit domains to the allowlist and keep tokens outside the repository.

