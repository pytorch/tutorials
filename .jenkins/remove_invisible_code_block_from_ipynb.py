import sys
from bs4 import BeautifulSoup

ipynb_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(ipynb_file_path, 'r', encoding='utf-8') as ipynb_file:
    ipynb_lines = ipynb_file.readlines()

ipynb_out_lines = []

for line in ipynb_lines:
    if not '%%%%%%INVISIBLE_CODE_BLOCK%%%%%%' in line:
        ipynb_out_lines.append(line)

with open(output_file_path, "w", encoding='utf-8') as output_file:
    for line in ipynb_out_lines:
        output_file.write(line)
