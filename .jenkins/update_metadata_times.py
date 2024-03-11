# Run this script to update the metadata.json file with the latest computation
# times from main and then make a PR to commit the change to the repo.
import os
import re
from urllib.request import urlopen
import requests
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
import rockset
import json
from pathlib import Path

REPO_ROOT = Path(__file__).absolute().parent.parent
OWNER = "pytorch"
REPO = "tutorials"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if GITHUB_TOKEN is None:
    raise RuntimeError("GITHUB_TOKEN is not set")
ROCKSET_API_KEY = os.environ.get("ROCKSET_API_KEY")
if ROCKSET_API_KEY is None:
    raise RuntimeError("ROCKSET_API_KEY is not set")


def git_api(
    url: str, params: Dict[str, str], type: str = "get", token: str = GITHUB_TOKEN
) -> Any:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    if type == "post":
        return requests.post(
            f"https://api.github.com{url}",
            data=json.dumps(params),
            headers=headers,
        ).json()
    elif type == "patch":
        return requests.patch(
            f"https://api.github.com{url}",
            data=json.dumps(params),
            headers=headers,
        ).json()
    else:
        return requests.get(
            f"https://api.github.com{url}",
            params=params,
            headers=headers,
        ).json()


def get_rockset_client():
    return rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )


def query_rockset(
    query: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = False
) -> List[Dict[str, Any]]:
    return get_rockset_client().sql(query, params=params).results


QUERY = """
select
    w.id,
    w.head_sha,
    ARRAY_AGG(j.id) as job_ids
from
    workflow_run w
    join workflow_job j on j.run_id = w.id
where
    w.name = 'Build tutorials'
    and w.conclusion = 'success'
    and ARRAY_CONTAINS(SPLIT(:shas, ','), w.head_sha)
group by
    w.id,
    w.head_sha
"""


def get_log(id):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/jobs/{id}/logs"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    with urlopen(Request(url, headers=headers)) as data:
        log_data = data.read().decode("utf-8")
    return log_data


def parse_log(log):
    res = {}
    for line in log.splitlines():
        rematch = re.search(" - ([^ ]+.py): +(\d*\.\d*) sec +\d+\.\d+ MB", line)
        if rematch:
            res[rematch.group(1)] = float(rematch.group(2))
    return res


def main():
    # Get list of commits on main
    j = git_api(
        f"/repos/{OWNER}/{REPO}/commits", params={"sha": "main", "per_page": 50}
    )
    shas = [x["sha"] for x in j]
    shas_str = ",".join(shas)

    # Query rockset for job ids corresponding to the commits
    r = query_rockset(QUERY, {"shas": shas_str})

    # Retrieve logs and parse for the computation times
    r_dict = {row["head_sha"]: row for row in r}
    durations = {}
    for sha in shas:
        try:
            row = r_dict.get(sha)
            if not row:
                continue
            job_ids = row["job_ids"]
            for job_id in job_ids:
                log = get_log(job_id)
                res = parse_log(log)
                for k, v in res.items():
                    if v > durations.get(k, 0):
                        durations[k] = v
            break
        except Exception as e:
            pass

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
