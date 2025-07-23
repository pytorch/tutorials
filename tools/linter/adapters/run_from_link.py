import argparse
import subprocess
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a formatter in a different repository.",
    )
    parser.add_argument(
        "--run-init",
        action="store_true",
        help="Run the initialization script specified by --init-name.",
    )
    parser.add_argument(
        "--run-lint",
        action="store_true",
        help="Run the linting script specified by --lint-name.",
    )
    parser.add_argument(
        "--init-name",
        help="Name of the initialization script.  This also serves as the filename.",
    )
    parser.add_argument(
        "--init-link",
        help="URL to download the initialization script from.",
    )
    parser.add_argument(
        "--lint-name",
        help="Name of the linting script.  This also serves as the filename.",
    )
    parser.add_argument(
        "--lint-link",
        help="URL to download the linting script from.",
    )

    parser.add_argument("args_for_file", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Skip the first -- if present
    if args.args_for_file and args.args_for_file[0] == "--":
        args.args_for_file = args.args_for_file[1:]
    return args


def download_file(url: str, location: Path) -> bytes:
    response = urllib.request.urlopen(url)
    content = response.read()
    location.write_bytes(content)
    return content


def main() -> None:
    args = parse_args()

    location = REPO_ROOT / ".lintbin" / "from_link" / "adapters"
    location.mkdir(parents=True, exist_ok=True)

    if args.lint_link:
        download_file(args.lint_link, location / args.lint_name)

    if args.init_link:
        download_file(args.init_link, location / args.init_name)

    if args.run_init:
        # Save the content to a file named after the name argument
        subprocess_args = ["python3", location / args.init_name] + args.args_for_file
        subprocess.run(subprocess_args, check=True)
    if args.run_lint:
        subprocess_args = ["python3", location / args.lint_name] + args.args_for_file
        subprocess.run(
            subprocess_args,
            check=True,
        )


if __name__ == "__main__":
    main()
