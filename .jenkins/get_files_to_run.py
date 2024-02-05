from typing import Any, Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
from remove_runnable_code import remove_runnable_code


# Calculate repo base dir
REPO_BASE_DIR = Path(__file__).absolute().parent.parent


def get_all_files() -> List[str]:
    sources = [x.relative_to(REPO_BASE_DIR) for x in REPO_BASE_DIR.glob("*_source/**/*.py") if 'data' not in x.parts]
    return sorted([str(x) for x in sources])


def read_metadata() -> Dict[str, Any]:
    # dcgan, ax_multiobjective_nas_tutorial, seq2seq_translation_tutorial don't
    # actually need a10g, but they're a lot faster on it
    with (REPO_BASE_DIR / ".jenkins" / "metadata.json").open() as fp:
        return json.load(fp)


def calculate_shards(all_files: List[str], num_shards: int = 20) -> List[List[str]]:
    sharded_files: List[Tuple[float, List[str]]] = [(0.0, []) for _ in range(num_shards)]
    metadata = read_metadata()

    def get_duration(file: str) -> int:
        # tutorials not listed in the metadata.json file usually take
        # <3min to run, so we'll default to 1min if it's not listed
        return metadata.get(file, {}).get("duration", 60)

    def get_needs_machine(file: str) -> Optional[str]:
        return metadata.get(file, {}).get("needs", None)

    def add_to_shard(i, filename):
        shard_time, shard_jobs = sharded_files[i]
        shard_jobs.append(filename)
        sharded_files[i] = (
            shard_time + get_duration(filename),
            shard_jobs,
        )

    all_other_files = all_files.copy()
    needs_multigpu = list(
        filter(lambda x: get_needs_machine(x) == "linux.16xlarge.nvidia.gpu", all_files,)
    )
    needs_a10g = list(
        filter(lambda x: get_needs_machine(x) == "linux.g5.4xlarge.nvidia.gpu", all_files,)
    )

    # Magic code for torchvision: for some reason, it needs to run after
    # beginner_source/basics/data_tutorial.py.  Very specifically:
    # https://github.com/pytorch/tutorials/blob/edff1330ca6c198e8e29a3d574bfb4afbe191bfd/beginner_source/basics/data_tutorial.py#L49-L60
    # So manually add them to the last shard
    add_to_shard(num_shards - 1, "beginner_source/basics/data_tutorial.py")
    add_to_shard(num_shards - 1, "intermediate_source/torchvision_tutorial.py")
    all_other_files.remove("beginner_source/basics/data_tutorial.py")
    all_other_files.remove("intermediate_source/torchvision_tutorial.py")

    for filename in needs_multigpu:
        # currently, the only job that has multigpu is the 0th worker,
        # so we'll add all the jobs that need this machine to the 0th worker
        add_to_shard(0, filename)
        all_other_files.remove(filename)
    for filename in needs_a10g:
        # currently, workers 1-5 use linux.g5.4xlarge.nvidia.gpu (sm86, A10G)
        min_shard_index = sorted(range(1, 6), key=lambda i: sharded_files[i][0])[
            0
        ]
        add_to_shard(min_shard_index, filename)
        all_other_files.remove(filename)
    sorted_files = sorted(all_other_files, key=get_duration, reverse=True,)

    for filename in sorted_files:
        min_shard_index = sorted(range(1, num_shards), key=lambda i: sharded_files[i][0])[
            0
        ]
        add_to_shard(min_shard_index, filename)
    return [x[1] for x in sharded_files]


def compute_files_to_keep(files_to_run: List[str]) -> List[str]:
    metadata = read_metadata()
    files_to_keep = list(files_to_run)
    for file in files_to_run:
        extra_files = metadata.get(file, {}).get("extra_files", [])
        files_to_keep.extend(extra_files)
    return files_to_keep


def remove_other_files(all_files, files_to_keep) -> None:

    for file in all_files:
        if file not in files_to_keep:
            remove_runnable_code(file, file)


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Select files to run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_WORKERS", "20")))
    parser.add_argument("--shard-num", type=int, default=int(os.environ.get("WORKER_ID", "1")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_files = get_all_files()
    files_to_run = calculate_shards(all_files, num_shards=args.num_shards)[args.shard_num - 1]
    if not args.dry_run:
        remove_other_files(all_files, compute_files_to_keep(files_to_run))
    stripped_file_names = [Path(x).stem for x in files_to_run]
    print(" ".join(stripped_file_names))


if __name__ == "__main__":
    main()
