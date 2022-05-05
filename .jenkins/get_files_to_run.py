from typing import List
from subprocess import run
import json
import os
from pathlib import Path
from remove_runnable_code import remove_runnable_code


def get_all_files(encoding="utf-8") -> List[str]:
    sources = [
        "beginner_source",
        "intermediate_source",
        "advanced_source",
        "recipes_source",
        "prototype_source",
    ]
    cmd = ["find"] + sources + ["-name", "*.py", "-not", "-path", "*/data/*"]

    return run(cmd, capture_output=True).stdout.decode(encoding).splitlines()


def calculate_shards(all_files, num_shards=20):
    metadata = json.load(open(".jenkins/metadata.json"))
    sharded_files = [(0.0, []) for _ in range(num_shards)]

    def get_duration(file):
        # tutorials not listed in the metadata.json file usually take
        # <3min to run, so we'll default to 1min if it's not listed
        return metadata.get(file, {}).get("duration", 60)

    def add_to_shard(i, filename):
        shard_time, shard_jobs = sharded_files[i]
        shard_jobs.append(filename)
        sharded_files[i] = (
            shard_time + get_duration(filename),
            shard_jobs,
        )

    needs_gpu_nvidia_small_multi = list(
        filter(lambda x: "needs" in metadata.get(x, {}), all_files)
    )
    for filename in needs_gpu_nvidia_small_multi:
        # currently, the only job that uses gpu.nvidia.small.multi is the 0th worker,
        # so we'll add all the jobs that need this machine to the 0th worker
        add_to_shard(0, filename)

    all_other_files = list(
        filter(lambda x: x not in needs_gpu_nvidia_small_multi, all_files)
    )

    sorted_files = sorted(all_other_files, key=get_duration, reverse=True,)

    for filename in sorted_files:
        min_shard_index = sorted(range(num_shards), key=lambda i: sharded_files[i][0])[
            0
        ]
        add_to_shard(min_shard_index, filename)
    return list(map(lambda x: x[1], sharded_files))


def remove_other_files(all_files, files_to_run):
    for file in all_files:
        if file not in files_to_run:
            remove_runnable_code(file, file)


def main():
    num_shards = int(os.environ.get("NUM_WORKERS", 20))
    shard_num = int(os.environ.get("WORKER_ID", 0))

    all_files = get_all_files()
    files_to_run = calculate_shards(all_files, num_shards=num_shards)[shard_num]
    remove_other_files(all_files, files_to_run)
    stripped_file_names = list(map(lambda x: Path(x).stem, files_to_run))
    print(" ".join(stripped_file_names))


if __name__ == "__main__":
    main()
