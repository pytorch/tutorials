import glob
from pathlib import Path
import shutil
import subprocess
import os
import time
from get_files_to_run import remove_other_files, compute_files_to_keep, calculate_shards, get_all_files
from validate_tutorials_built import NOT_RUN

def print_files(files):
    print(f"Files to run ({len(files)}):")
    for file in files:
        print(f"- {file}")


def main() -> None:
    all_files = get_all_files()
    files_to_run = calculate_shards(all_files, num_shards=15)[int(os.environ.get("WORKER_ID", "1")) - 1]
    files_to_run = [x for x in files_to_run if x not in [f"{f}.py" for f in NOT_RUN]]

    os.makedirs("/tmp/docs_to_zip", exist_ok=True)

    env = os.environ.copy()
    for file in files_to_run:
        print(f"Running {file}")
        start = time.time()
        remove_other_files(all_files, compute_files_to_keep([file]))
        stem = Path(file).stem
        env["RUNTHIS"] = stem
        env["FILES_TO_RUN"] = stem

        subprocess.check_output(["make", "download"], env=env)
        result = subprocess.check_output(["make", "html"], env=env)
        print(result.decode("utf-8"))
        subprocess.check_output(["make", "postprocess"], env=env)
        print("Done running")
        for file in glob.glob(f"docs/**/*", recursive=True):
            if stem in file:
                relative_path = Path(os.path.relpath(file, "docs"))
                print(relative_path)
                print(relative_path.parent)
                os.makedirs(os.path.dirname(f"/tmp/docs_to_zip/{relative_path}"), exist_ok=True)
                shutil.copy(file, f"/tmp/docs_to_zip/{relative_path}")
        subprocess.check_output(["git", "reset", "--hard", "HEAD"])
        subprocess.check_output(["git", "clean", "-f", "-d"])
        print(f"Done with {file} in {time.time() - start:.2f} seconds")

    shutil.rmtree("_build")
    os.makedirs("_build", exist_ok=True)
    shutil.move("/tmp/docs_to_zip", "_build/html")

if __name__ == "__main__":
    main()
