#!/usr/bin/env python3
"""
Standalone script to perform a full WebArena reset.

Restores ALL websites (shopping, shopping_admin, reddit, gitlab, etc.) to their
initial database state via the WA_FULL_RESET endpoint. Takes 200-500 seconds.

NOTE: The reset endpoint is global — it resets all sites simultaneously.
Do NOT run this while tasks are executing on any site.

Usage:
    python reset_webarena.py

    # Or source env first if not already exported:
    source env_setup.txt && python reset_webarena.py
"""

import os
import sys
import time
from pathlib import Path

import requests


def load_webarena_env():
    env_file = Path(__file__).parent / "env_setup.txt"
    if not env_file.exists():
        print(f"Warning: env_setup.txt not found at {env_file}")
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if '$BASE_URL' in value and 'BASE_URL' in os.environ:
                    value = value.replace('$BASE_URL', os.environ['BASE_URL'])
                os.environ[key] = value


def main():
    load_webarena_env()

    base_url = os.environ.get('WA_FULL_RESET', '').rstrip('/')
    if not base_url:
        print("ERROR: WA_FULL_RESET is not set. Check env_setup.txt.")
        sys.exit(1)

    reset_url = f"{base_url}/reset"
    status_url = f"{base_url}/status"

    print("=" * 60)
    print("WebArena Full Reset")
    print(f"Endpoint: {base_url}")
    print("Resetting ALL websites to initial state...")
    print("Expected time: 200-500 seconds")
    print("=" * 60)

    # Trigger the reset
    response = requests.get(reset_url)
    if response.status_code == 200:
        print("Reset started.")
    elif response.status_code == 418:
        print("Reset was already running, waiting for it to finish...")
    else:
        print(f"ERROR: Reset request failed ({response.status_code}): {response.text}")
        sys.exit(1)

    # Poll until ready
    poll_interval = 20  # seconds
    timeout = 10 * 60  # 10 minutes
    start = time.time()
    while True:
        response = requests.get(status_url)
        if response.status_code != 200:
            print(f"ERROR: Status request failed ({response.status_code}): {response.text}")
            sys.exit(1)
        if response.text == "Ready for duty!":
            break
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"ERROR: Reset timed out after {elapsed:.0f}s")
            sys.exit(1)
        print(f"  Still resetting... ({elapsed:.0f}s elapsed)")
        time.sleep(poll_interval)

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Reset complete in {elapsed:.0f}s. All websites restored to initial state.")
    print("=" * 60)


if __name__ == "__main__":
    main()
