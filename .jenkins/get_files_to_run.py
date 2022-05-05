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

    sorted_files = sorted(all_files, key=lambda x: metadata.get(x, {}).get("duration", 1), reverse=True)
    for filename in sorted_files:
        min_shard_index = sorted(range(num_shards), key=lambda i: sharded_files[i][0])[0]
        curr_shard_time, curr_shard_jobs = sharded_files[min_shard_index]
        curr_shard_jobs.append(filename)
        sharded_files[min_shard_index] = (
            curr_shard_time + metadata.get(filename, {}).get("duration", 1),
            curr_shard_jobs,
        )
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
