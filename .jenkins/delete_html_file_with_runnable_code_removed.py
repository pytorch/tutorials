import sys
import os

html_file_path = sys.argv[1]

with open(html_file_path, 'r', encoding='utf-8') as html_file:
    html = html_file.read()

if "%%%%%%RUNNABLE_CODE_REMOVED%%%%%%" in html:
    print("Removing " + html_file_path)
    os.remove(html_file_path)
