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


def main():
    num_shards = int(os.environ.get("NUM_WORKERS", 20))
    shard_num = int(os.environ.get("WORKER_ID", 0))

    all_files = get_all_files()
    files_to_run = []
    for i, name in enumerate(all_files):
        if i % num_shards == shard_num:
            files_to_run.append(name)
        else:
            remove_runnable_code(name, name)
    stripped_file_names = list(map(lambda x: Path(x).stem, files_to_run))
    print(" ".join(stripped_file_names))


if __name__ == "__main__":
    main()
