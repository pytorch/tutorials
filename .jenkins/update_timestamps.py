import re
import sys
import os
import subprocess

def get_last_commit_timestamp_for_file(file_path: str) -> str:
    """Get the last commit timestamp for a file.

    Args:
        file_path (str): Path to file

    Returns:
        str: Last committed timestamp string
    """
    git_command = ["git", "log", "-1", "--format=%at", "--", file_path]
    timestamp = subprocess.check_output(git_command).decode().strip()

    if not timestamp:
        # If there is no git commit history, use last modified date
        timestamp = str(int(os.path.getmtime(file_path)))

    date_command = ["date", "-d", "@" + timestamp, "+%I:%M %p, %B %d, %Y"]
    return subprocess.check_output(date_command).decode().strip()

def update_timestamp(file_path: str):
    """Adds a timestamp of the most recent time the file was edited.

    Args:
        file_path (str): Path to file
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Get current timestamp
    timestamp = get_last_commit_timestamp_for_file(file_path)
    timestamp_line = f'{"# " if file_path.endswith("py") else ""}**Updated:** *{timestamp}*\n'
    timestamp_pattern = r'\*\*Updated:\*\*\s\**\d{1,2}:\d{2} [AP]M, \w+ \d{1,2}, \d{4}\*'

    if not lines:
        lines = [timestamp_line]
    else:
        i = len(lines) - 1
        while i > 0 and not lines[i].strip():
            i -= 1

        if re.search(timestamp_pattern, lines[i]):
            lines[i] = timestamp_line
        else:
            lines.append('\n\n' + timestamp_line)
    
    # Write updated lines back to file
    with open(file_path, 'w') as file:
        file.writelines(lines)


file_path = sys.argv[1]
update_timestamp(file_path)