import shutil
import subprocess
import os
import glob

def main() -> None:
    files_to_run = os.environ["FILES_TO_RUN"]
    env = os.environ.copy()
    for file in files_to_run.split(" "):
        print(f"Running {file}")
        env["RUNTHIS"] = file
        subprocess.check_output(["make", "html", f"BUILDDIR=_build/{file}"], env=env)
        files = glob.glob(f"_build/{file}/**/*", recursive=True)
        for gen_file in files:
            if file in str(gen_file):
                rel_path = os.path.relpath(gen_file, f"_build/{file}")
                print(gen_file, f"_build/{rel_path}")
                shutil.copy(gen_file, f"_build/{rel_path}")

if __name__ == "__main__":
    main()
