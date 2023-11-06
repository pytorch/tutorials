"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Usage: python ipynb_to_gallery.py <notebook.ipynb>
Dependencies:
pypandoc: install using `pip install pypandoc`
"""
import pypandoc as pdoc
import json

def convert_ipynb_to_gallery(file_name):
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:  
            assert cell['cell_type'] == 'markdown', \
                'First cell has to be markdown'

            md_source = ''.join(cell['source'])
            rst_source = pdoc.convert_text(md_source, 'rst', 'md')
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                commented_source = ''.join(['# ' + x for x in
                                              rst_source.split('\n')])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source

    python_file = python_file.replace("\n%", "\n# %")
    open(file_name.replace('.ipynb', '.py'), 'w').write(python_file)

if __name__ == '__main__':
    import sys
    convert_ipynb_to_gallery(r"C:\Users\meast\tiatoolbox\examples\05-patch-prediction.ipynb")