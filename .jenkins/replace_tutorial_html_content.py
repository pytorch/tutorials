import sys

noplot_html_file_path = sys.argv[1]
hasplot_html_file_path = sys.argv[2]
output_html_file_path = sys.argv[3]

from bs4 import BeautifulSoup
with open(noplot_html_file_path, 'r', encoding='utf-8') as noplot_html_file:
  noplot_html = noplot_html_file.read()
with open(hasplot_html_file_path, 'r', encoding='utf-8') as hasplot_html_file:
  hasplot_html = hasplot_html_file.read()

noplot_html_soup = BeautifulSoup(noplot_html, 'html.parser')
elems = noplot_html_soup.find_all("div", {"class": "sphx-glr-example-title"})
if len(elems) == 0:
  print("No match found, not replacing HTML content in "+noplot_html_file_path)
elif len(elems) == 1:
  print("Match found in "+noplot_html_file_path+". Replacing its content.")
  elem = elems[0]
  elem.replace_with(BeautifulSoup(hasplot_html, 'html.parser').find_all("div", {"class": "sphx-glr-example-title"})[0])
  with open(output_html_file_path, "w", encoding='utf-8') as output_html_file:
    output_html_file.write(str(noplot_html_soup))
else:
  raise Exception("Found more than one match in "+noplot_html_file_path+". Aborting.")
