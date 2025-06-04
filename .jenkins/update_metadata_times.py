# Run this script locally to update the metadata.json file with the latest
# computation times from main and then make a PR to commit the change to the
# repo.
import os
import re
import requests
from typing import  List
import json
from pathlib import Path

REPO_ROOT = Path(__file__).absolute().parent.parent
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if GITHUB_TOKEN is None:
    raise RuntimeError("GITHUB_TOKEN is not set")


def get_log(id: str) -> str:
    url = f"https://api.github.com/repos/pytorch/tutorials/actions/jobs/{id}/logs"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    log_data = res.text
    return log_data


def parse_log(log: str) -> dict:
    res = {}
    for line in log.splitlines():
        rematch = re.search(" - ([^ ]+.py): +(\d*\.\d*) sec +\d+\.\d+ MB", line)
        if rematch:
            res[rematch.group(1)] = float(rematch.group(2))
    return res


def get_log_ids() -> List[str]:
    url = f"https://api.github.com/repos/pytorch/tutorials/actions/workflows/build-tutorials.yml/runs?branch=main&status=completed&per_page=100"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    runs = response.json().get("workflow_runs", [])
    for run in runs:
        jobs_url = run.get("jobs_url")
        if not jobs_url:
            continue
        jobs_response = requests.get(jobs_url, headers=headers)
        jobs_response.raise_for_status()
        print(json.dumps(jobs_response.json(), indent=2))
        jobs =jobs_response.json().get("jobs", [])
        return [job["id"] for job in jobs]
    raise RuntimeError("No jobs found for the given SHA")

def main():
    log_ids = get_log_ids()
    durations = {}
    for log_id in log_ids:
        log = get_log(log_id)
        res = parse_log(log)
        for k, v in res.items():
            if v > durations.get(k, 0):
                durations[k] = v

    # Write back to metadata.json
    with open(REPO_ROOT / ".jenkins/metadata.json", "r") as f:
        metadata = json.load(f)
    for k, v in durations.items():
        if k not in metadata:
            metadata[k] = {}
        metadata[k]["duration"] = v
    with open(REPO_ROOT / ".jenkins/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
