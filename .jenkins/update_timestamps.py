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

    author_line_index = -1

    # Find the index of the author line and extract author's name and GitHub link
    for i, line in enumerate(lines):
        if re.search(r'(Author|Authors).*?:', line):
            author_line_index = i
            break

    # Get current timestamp
    timestamp = get_last_commit_timestamp_for_file(file_path)
    timestamp_line = f'**Updated:** *{timestamp}*\n'

    # If author line is found, add timestamp below it
    if author_line_index != -1:
        
        if lines[author_line_index].startswith('#'):
            # We can assume we need a #, too
            timestamp_line = '# ' + timestamp_line
            
        updated_lines = lines[:author_line_index + 1]
        # Check if the timestamp line exists below the author line or if there are only blank lines between them
        if author_line_index + 1 < len(lines) and (lines[author_line_index + 1].strip() == '' or re.search(r'\*\*Updated:\*\*\s\**\d{1,2}:\d{2} [AP]M, \w+ \d{1,2}, \d{4}\*', lines[author_line_index + 1])):
            # If timestamp line exists or there are only blank lines, update it
            i = author_line_index + 1
            while i < len(lines) and lines[i].strip() == '':
                # Find first non-empty line after Author
                updated_lines.append(lines[i])
                i += 1

            if re.search(r'\*\*Updated:\*\*\s\**\d{1,2}:\d{2} [AP]M, \w+ \d{1,2}, \d{4}\*', lines[i]):
                updated_lines.append(timestamp_line)
            else:
                updated_lines[author_line_index + 1] = timestamp_line
                if i == author_line_index + 2: updated_lines.append('\n')

            updated_lines.extend(lines[i+1:])
        else:
            # If timestamp line does not exist and there are no blank lines, add it below author line
            updated_lines += [timestamp_line, '\n'] + lines[author_line_index + 1:]
    else:
        # If author line is not found, add timestamp to the last line
        updated_lines = lines

        if file_path.endswith('.py'): timestamp_line = '# ' + timestamp_line

        i = len(lines) - 1
        while i >= 0 and lines[i].strip() == '':
            # Go to the last non-blank line, check if it is the timestamp
            i -= 1

        if i >= 0 and re.search(r'\*\*Updated:\*\*\s\**\d{1,2}:\d{2} [AP]M, \w+ \d{1,2}, \d{4}\*', lines[i]):
            updated_lines[i] = timestamp_line
        else:
            updated_lines.append(f'\n\n{timestamp_line}')
    
    # Write updated lines back to file
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)


file_path = sys.argv[1]
update_timestamp(file_path)