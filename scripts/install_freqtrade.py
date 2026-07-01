from __future__ import annotations

import logging
import shutil
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


def check_freqtrade_available() -> bool:
    return shutil.which("freqtrade") is not None


def print_install_hint() -> None:
    if check_freqtrade_available():
        LOGGER.info("Freqtrade CLI is available.")
        return
    LOGGER.warning(
        "Freqtrade CLI is not installed. Install locally when needed: "
        "pip install freqtrade, or follow https://www.freqtrade.io/en/stable/installation/"
    )


def run_freqtrade_dry_command(args: list[str]) -> int:
    if not check_freqtrade_available():
        print_install_hint()
        return 2
    command = ["freqtrade", *args]
    LOGGER.info("Running local freqtrade command: %s", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_install_hint()
    sys.exit(0)

