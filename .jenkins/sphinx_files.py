import subprocess
import os


def main() -> None:
    files_to_run = os.environ["FILES_TO_RUN"]
    env = os.environ.copy()
    for file in files_to_run.split(" "):
        print(f"Running {file}")
        env["RUN_ONLY"] = file
        subprocess.check_output(["make", "html"], env=env)


if __name__ == "__main__":
    main()
