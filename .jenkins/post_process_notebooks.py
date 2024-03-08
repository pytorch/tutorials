import nbformat as nbf
import os
import re

# Pattern to search ``` {.python .jupyter-code-cell}
pattern = re.compile(r'(.*?)``` {.python .jupyter-code-cell}\n\n(from IPython.display import display, HTML\npython_code = """\n.*?\n"""\ndisplay\(HTML\(python_code\)\))\n```(.*)', re.DOTALL)

def get_gallery_dirs(conf_path):
    """
    Execute the conf.py file and return the gallery directories.
    This is needed to make sure the script runs through all
    dirs.
    """
    namespace = {}
    exec(open(conf_path).read(), namespace)
    sphinx_gallery_conf = namespace['sphinx_gallery_conf']
    print(f"Processing directories: {', '.join(sphinx_gallery_conf['gallery_dirs'])}")
    return sphinx_gallery_conf['gallery_dirs']

def process_video_cell(notebook_path):
    """
    This function finds the code blocks with the
    "``` {.python .jupyter-code-cell}" code bocks and slices them
    into a separe code cell (instead of markdown) which allows to
    load the video in the notebook. The rest of the content is placed
    in a new markdown cell.
    """
    print(f'Processing file: {notebook_path}')
    notebook = nbf.read(notebook_path, as_version=4)

    # Iterate over markdown cells
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'markdown':
            match = pattern.search(cell.source)
            if match:
                print(f'Match found in cell {i}: {match.group(0)[:100]}...')
                # Extract the parts before and after the video code block
                before_html_block = match.group(1)
                code_block = match.group(2)

                # Add a comment to run the cell to display the video 
                code_block = "# Run this cell to load the video\n" + code_block
                # Create a new code cell
                new_code_cell = nbf.v4.new_code_cell(source=code_block)

                # Replace the original markdown cell with the part before the code block
                cell.source = before_html_block

                # Insert the new code cell after the current one
                notebook.cells.insert(i+1, new_code_cell)

                # If there is content after the HTML code block, create a new markdown cell
                if len(match.group(3).strip()) > 0:
                    after_html_block = match.group(3)
                    new_markdown_cell = nbf.v4.new_markdown_cell(source=after_html_block)
                    # Create a new markdown cell and add the content after code block there
                    notebook.cells.insert(i+2, new_markdown_cell)

            else:
                # Remove ```{=html} from the code block
                cell.source = remove_html_tag(cell.source)

    nbf.write(notebook, notebook_path)

def remove_html_tag(content):
    """
    Pandoc adds an extraneous ```{=html} ``` to raw HTML blocks which
    prevents it from rendering correctly. This function removes
    ```{=html} that we don't need.
    """
    content = re.sub(r'```{=html}\n<div', '<div', content)
    content = re.sub(r'">\n```', '">', content)
    content = re.sub(r'<\/div>\n```', '</div>\n', content)
    content = re.sub(r'```{=html}\n</div>\n```', '</div>\n', content)
    content = re.sub(r'```{=html}', '', content)
    content = re.sub(r'</p>\n```', '</p>', content)
    return content

def walk_dir(notebook_dir):
    """
    Walk the dir and process all notebook files in
    the gallery directory and its subdirectories.
    """
    for root, dirs, files in os.walk(notebook_dir):
        for filename in files:
            if filename.endswith('.ipynb'):
                process_video_cell(os.path.join(root, filename))

def main():
    conf_path = './conf.py'
    for notebook_dir in get_gallery_dirs(conf_path):
        walk_dir(notebook_dir)

if __name__ == "__main__":
    main()
