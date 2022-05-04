#!/usr/bin/env python3

# regenrates config.yml based on config.yml.in

from copy import deepcopy
import os.path

import jinja2
import yaml
from jinja2 import select_autoescape

WORKFLOWS_JOBS_PR = {"filters": {"branches": {"ignore": ["master"]}}}

WORKFLOWS_JOBS_MASTER = {
    "context": "org-member",
    "filters": {"branches": {"only": ["master"]}},
}


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(
        yaml.dump(data_list, default_flow_style=False).splitlines()
    )


def jobs(pr_or_master, num_workers=20, indentation=2):
    w = {}
    needs_gpu_nvidia_small_multi = [3, 9, 12, 13]
    needs_gpu_nvidia_medium = [14, 15]
    w[f"pytorch_tutorial_{pr_or_master}_build_manager"] = {
        "<<": "*pytorch_tutorial_build_manager_defaults"
    }
    for i in range(num_workers):
        d = {"<<": "*pytorch_tutorial_build_worker_defaults"}
        if i in needs_gpu_nvidia_small_multi:
            d["resource_class"] = "gpu.nvidia.small.multi"
        if i in needs_gpu_nvidia_medium:
            d["resource_class"] = "gpu.nvidia.medium"
        w[f"pytorch_tutorial_{pr_or_master}_build_worker_{i}"] = d

    return indent(indentation, w).replace("'", "")


def workflows_jobs(pr_or_master, indentation=6, num_workers=20):
    w = []
    d = deepcopy(WORKFLOWS_JOBS_PR if pr_or_master == "pr" else WORKFLOWS_JOBS_MASTER)

    for i in range(num_workers):
        w.append({f"pytorch_tutorial_{pr_or_master}_build_worker_{i}": deepcopy(d)})

    d["requires"] = [
        f"pytorch_tutorial_{pr_or_master}_build_worker_{i}" for i in range(num_workers)
    ]
    w.append({f"pytorch_tutorial_{pr_or_master}_build_manager": deepcopy(d)})
    return indent(indentation, w)


def windows_jobs(indentation=2, num_workers=4):
    w = {}
    for i in range(num_workers):
        w[f"pytorch_tutorial_windows_pr_build_worker_{i}"] = {
            "<<": "*pytorch_windows_build_worker"
        }
        w[f"pytorch_tutorial_windows_master_build_worker_{i}"] = {
            "<<": "*pytorch_windows_build_worker"
        }
    return indent(indentation, w).replace("'", "")


def windows_workflows_jobs(indentation=6, num_workers=4):
    w = []
    d = WORKFLOWS_JOBS_PR
    for i in range(num_workers):
        w.append({f"pytorch_tutorial_windows_pr_build_worker_{i}": deepcopy(d)})

    d = WORKFLOWS_JOBS_MASTER
    for i in range(num_workers):
        w.append({f"pytorch_tutorial_windows_master_build_worker_{i}": deepcopy(d)})

    return ("\n#").join(indent(indentation, w).splitlines())


if __name__ == "__main__":

    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        keep_trailing_newline=True,
    )
    with open(os.path.join(d, "config.yml"), "w") as f:
        f.write(
            env.get_template("config.yml.in").render(
                jobs=jobs,
                workflows_jobs=workflows_jobs,
                windows_jobs=windows_jobs,
                windows_workflows_jobs=windows_workflows_jobs,
            )
        )
