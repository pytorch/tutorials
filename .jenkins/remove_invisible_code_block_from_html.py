import sys
from bs4 import BeautifulSoup

html_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(html_file_path, 'r', encoding='utf-8') as html_file:
    html = html_file.read()
html_soup = BeautifulSoup(html, 'html.parser')

elems = html_soup.find_all("div", {"class": "highlight-default"})
for elem in elems:
    if "%%%%%%INVISIBLE_CODE_BLOCK%%%%%%" in str(elem):
        elem.decompose()

with open(output_file_path, "w", encoding='utf-8') as output_file:
    output_file.write(str(html_soup))
