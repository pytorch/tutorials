import sys
from bs4 import BeautifulSoup

rst_txt_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(rst_txt_file_path, 'r', encoding='utf-8') as rst_txt_file:
    rst_txt = rst_txt_file.read()

splits = rst_txt.split('.. code-block:: default\n\n\n    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%\n')
if len(splits) == 2:
    code_before_invisible_block = splits[0]
    code_after_invisible_block = splits[1].split('    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%\n')[1]
    rst_txt_out = code_before_invisible_block + code_after_invisible_block
else:
    rst_txt_out = rst_txt

with open(output_file_path, "w", encoding='utf-8') as output_file:
    output_file.write(rst_txt_out)
