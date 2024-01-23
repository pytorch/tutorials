import nbformat as nbf
import os
import re

def get_gallery_dirs(conf_path):
    """Execute the conf.py file and return the gallery directories."""
    namespace = {}
    exec(open(conf_path).read(), namespace)
    sphinx_gallery_conf = namespace['sphinx_gallery_conf']
    print(f"Processing directories: {', '.join(sphinx_gallery_conf['gallery_dirs'])}")
    return sphinx_gallery_conf['gallery_dirs']

def process_notebook(notebook_path):
    """Read and process a notebook file."""
    print(f'Processing file: {notebook_path}')
    notebook = nbf.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            cell.source = process_content(cell.source)
    nbf.write(notebook, notebook_path)

def process_content(content):
    """Remove extra syntax from the content of a Markdown cell."""
    content = re.sub(r'```{=html}\n<div', '<div', content)
    content = re.sub(r'">\n```', '">', content)
    content = re.sub(r'<\/div>\n```', '</div>\n', content)
    content = re.sub(r'```{=html}\n</div>\n```', '</div>\n', content)
    content = re.sub(r'```{=html}', '', content)
    content = re.sub(r'</p>\n```', '</p>', content)
    return content

def process_directory(notebook_dir):
    """Process all notebook files in a directory and its subdirectories."""
    for root, dirs, files in os.walk(notebook_dir):
        for filename in files:
            if filename.endswith('.ipynb'):
                process_notebook(os.path.join(root, filename))

def main():
    """Main function to process all directories specified in the conf.py file."""
    conf_path = 'conf.py'
    for notebook_dir in get_gallery_dirs(conf_path):
        process_directory(notebook_dir)

if __name__ == "__main__":
    main()
