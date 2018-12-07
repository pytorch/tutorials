import sys
from bs4 import BeautifulSoup

py_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(py_file_path, 'r', encoding='utf-8') as py_file:
    py_lines = py_file.readlines()

py_out_lines = []

in_invisible_block = False
for line in py_lines:
    if not in_invisible_block:
        if '%%%%%%INVISIBLE_CODE_BLOCK%%%%%%' in line:
            in_invisible_block = True
        else:
            py_out_lines.append(line)
    else:
        if '%%%%%%INVISIBLE_CODE_BLOCK%%%%%%' in line:
            in_invisible_block = False

with open(output_file_path, "w", encoding='utf-8') as output_file:
    for line in py_out_lines:
        output_file.write(line)
