#!/usr/bin/env python3

# regenrates config.yml based on config.yml.in

from copy import deepcopy
import os.path

import jinja2
import yaml
from jinja2 import select_autoescape

WORKFLOWS_JOBS_PR = {"filters": {"branches": {"ignore": ["master", "main"]}}}

WORKFLOWS_JOBS_TRUNK = {
    "context": "org-member",
    "filters": {"branches": {"only": ["master", "main"]}},
}


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(
        yaml.dump(data_list, default_flow_style=False).splitlines()
    )


def jobs(pr_or_trunk, num_workers=20, indentation=2):
    jobs = {}

    # all tutorials that need gpu.nvidia.small.multi machines will be routed by
    # get_files_to_run.py to 0th worker, similarly for gpu.nvidia.large and the
    # 1st worker
    needs_gpu_nvidia_small_multi = [0]
    needs_gpu_nvidia_large = [1]
    jobs[f"pytorch_tutorial_{pr_or_trunk}_build_manager"] = {
        "<<": "*pytorch_tutorial_build_manager_defaults"
    }
    for i in range(num_workers):
        job_info = {"<<": "*pytorch_tutorial_build_worker_defaults"}
        if i in needs_gpu_nvidia_small_multi:
            job_info["resource_class"] = "gpu.nvidia.small.multi"
        if i in needs_gpu_nvidia_large:
            job_info["resource_class"] = "gpu.nvidia.large"
        jobs[f"pytorch_tutorial_{pr_or_trunk}_build_worker_{i}"] = job_info

    return indent(indentation, jobs).replace("'", "")


def workflows_jobs(pr_or_trunk, indentation=6, num_workers=20):
    jobs = []
    job_info = deepcopy(
        WORKFLOWS_JOBS_PR if pr_or_trunk == "pr" else WORKFLOWS_JOBS_TRUNK
    )

    for i in range(num_workers):
        jobs.append(
            {f"pytorch_tutorial_{pr_or_trunk}_build_worker_{i}": deepcopy(job_info)}
        )

    job_info["requires"] = [
        f"pytorch_tutorial_{pr_or_trunk}_build_worker_{i}" for i in range(num_workers)
    ]
    jobs.append({f"pytorch_tutorial_{pr_or_trunk}_build_manager": deepcopy(job_info)})
    return indent(indentation, jobs)


def windows_jobs(indentation=2, num_workers=4):
    jobs = {}
    for i in range(num_workers):
        jobs[f"pytorch_tutorial_windows_pr_build_worker_{i}"] = {
            "<<": "*pytorch_windows_build_worker"
        }
        jobs[f"pytorch_tutorial_windows_trunk_build_worker_{i}"] = {
            "<<": "*pytorch_windows_build_worker"
        }
    return indent(indentation, jobs).replace("'", "")


def windows_workflows_jobs(indentation=6, num_workers=4):
    jobs = []
    job_info = WORKFLOWS_JOBS_PR
    for i in range(num_workers):
        jobs.append(
            {f"pytorch_tutorial_windows_pr_build_worker_{i}": deepcopy(job_info)}
        )

    job_info = WORKFLOWS_JOBS_TRUNK
    for i in range(num_workers):
        jobs.append(
            {f"pytorch_tutorial_windows_trunk_build_worker_{i}": deepcopy(job_info)}
        )

    return ("\n#").join(indent(indentation, jobs).splitlines())


if __name__ == "__main__":

    directory = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(directory),
        lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        keep_trailing_newline=True,
    )
    with open(os.path.join(directory, "config.yml"), "w") as f:
        f.write(
            env.get_template("config.yml.in").render(
                jobs=jobs,
                workflows_jobs=workflows_jobs,
                windows_jobs=windows_jobs,
                windows_workflows_jobs=windows_workflows_jobs,
            )
        )
