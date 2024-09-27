import re
import sys
import os
import datetime
import subprocess


def get_last_commit_timestamp_for_file(file_path: str) -> str:
    """Get the last commit timestamp for a file.

    Args:
        file_path (str): Path to file

    Returns:
        str: Last committed timestamp string
    """

    git_command = [
        "git",
        "log",
        "-1",
        "--date=format:%B %d, %Y",
        "--format=%at",
        "--",
        file_path,
    ]
    timestamp = subprocess.check_output(git_command).decode().strip()

    if not timestamp:
        # If there is no git commit history, use last modified date
        timestamp = str(int(os.path.getmtime(file_path)))

    dt = datetime.datetime.fromtimestamp(int(timestamp), tz=datetime.timezone.utc)
    return dt.strftime("%I:%M %p, %B %d, %Y")


def update_timestamp(file_path: str):
    """Adds a timestamp of the most recent time the file was edited.

    Args:
        file_path (str): Path to file
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Get current timestamp
    timestamp = get_last_commit_timestamp_for_file(file_path)
    if not timestamp:
        return

    timestamp_line = (
        f'{"# " if file_path.endswith("py") else ""}_Last Updated: {timestamp}_\n'
    )
    timestamp_pattern = r"_Last Updated:\s.+_"

    if not lines:
        lines = [timestamp_line]
    else:
        i = len(lines) - 1
        while i > 0 and not lines[i].strip():
            i -= 1

        if re.search(timestamp_pattern, lines[i]):
            lines[i] = timestamp_line
        else:
            lines.append("\n\n" + timestamp_line)

    # Write updated lines back to file
    with open(file_path, "w") as file:
        file.writelines(lines)


file_path = sys.argv[1]
update_timestamp(file_path)
