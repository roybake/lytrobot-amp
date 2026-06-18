#!/usr/bin/env python3
"""Tag WandB runs that have empty history with `empty_run`.

This script uses the wandb public API to list runs and the REST API
to PATCH run metadata (tags). It reads the API key from ~/.netrc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import wandb
from netrc import netrc


def get_wandb_api_key() -> str | None:
    try:
        auth = netrc().authenticators("api.wandb.ai")
        if auth is None:
            return None
        _, _, password = auth
        return password
    except FileNotFoundError:
        return None


def main():
    api_key = get_wandb_api_key()
    if not api_key:
        print("No WandB API key found in ~/.netrc. Run `wandb login` first.")
        sys.exit(1)

    api = wandb.Api()
    # Replace with your project path if different
    project_path = "lhe216002-jilin-university/lytrobot-amp"
    try:
        runs = api.runs(project_path, per_page=200)
    except Exception as e:
        print("Failed to list runs:", e)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    entity, project = project_path.split("/")
    patched = 0
    from urllib.parse import urlparse
    for r in runs:
        rows = r.history(samples=1, pandas=False)
        if len(rows) == 0:
            run_id = r.id
            # Derive host from run.url (handles wandb.ai vs enterprise hosts)
            parsed = urlparse(r.url)
            host = f"{parsed.scheme}://{parsed.netloc}"
            urls = [
                f"{host}/api/v1/projects/{entity}/{project}/runs/{run_id}",
                f"{host}/api/v1/runs/{entity}/{project}/{run_id}",
                f"{host}/api/projects/{entity}/{project}/runs/{run_id}",
            ]
            tags = list(r.tags or [])
            if "empty_run" in tags:
                print(f"already tagged: {run_id} {r.name}")
                continue
            tags.append("empty_run")
            body = {"tags": tags}
            # Prefer using the wandb Python client to update tags
            try:
                tags = list(r.tags or [])
                if "empty_run" in tags:
                    print(f"already tagged: {run_id} {r.name}")
                    continue
                tags.append("empty_run")
                r.tags = tags
                r.update()
                print(f"tagged: {run_id} {r.name}")
                patched += 1
            except Exception as e:
                print(f"failed updating tags for {run_id}: {e}")

    print(f"done, patched={patched}")


if __name__ == "__main__":
    main()
