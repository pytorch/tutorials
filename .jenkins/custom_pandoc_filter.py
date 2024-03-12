from pandocfilters import toJSONFilter, Div, RawBlock, Para, Str, Space, Link, Code, CodeBlock
import markdown
import html

def to_markdown(item, skip_octicon=False):
    # A handler function to process strings, links, code, and code
    # blocks
    if item['t'] == 'Str':
        return item['c']
    elif item['t'] == 'Space':
        return ' '
    elif item['t'] == 'Link':
        link_text = ''.join(to_markdown(i, skip_octicon) for i in item['c'][1])
        return f'<a href="{item["c"][2][0]}">{link_text}</a>'
    elif item['t'] == 'Code':
        # Need to remove icticon as they don't render in .ipynb
        if any(value == 'octicon' for key, value in item['c'][0][2]):
            return ''
        else:
            # Escape the code and wrap it in <code> tags
            return f'<code>{html.escape(item["c"][1])}</code>'
    elif item['t'] == 'CodeBlock':
        # Escape the code block and wrap it in <pre><code> tags
        return f'<pre><code>{html.escape(item["c"][1])}</code></pre>'
    else:
        return ''

def process_admonitions(key, value, format, meta):
    # Replace admonitions with proper HTML.
    if key == 'Div':
        [[ident, classes, keyvals], contents] = value
        if 'note' in classes:
            color = '#54c7ec'
            label = 'NOTE:'
        elif 'tip' in classes:
            color = '#6bcebb'
            label = 'TIP:'
        elif 'warning' in classes:
            color = '#e94f3b'
            label = 'WARNING:'
        else:
            return

        note_content = []
        for block in contents:
            if block.get('t') == 'Para':
                for item in block['c']:
                    if item['t'] == 'Str':
                        note_content.append(Str(item['c']))
                    elif item['t'] == 'Space':
                        note_content.append(Space())
                    elif item['t'] == 'Link':
                        note_content.append(Link(*item['c']))
                    elif item['t'] == 'Code':
                        note_content.append(Code(*item['c']))
            elif block.get('t') == 'CodeBlock':
                note_content.append(CodeBlock(*block['c']))

        note_content_md = ''.join(to_markdown(item) for item in note_content)
        html_content = markdown.markdown(note_content_md)

        return [{'t': 'RawBlock', 'c': ['html', f'<div style="background-color: {color}; color: #fff; font-weight: 700; padding-left: 10px; padding-top: 5px; padding-bottom: 5px">{label}</div>']}, {'t': 'RawBlock', 'c': ['html', '<div style="background-color: #f3f4f7; padding-left: 10px; padding-top: 10px; padding-bottom: 10px; padding-right: 10px">']}, {'t': 'RawBlock', 'c': ['html', html_content]}, {'t': 'RawBlock', 'c': ['html', '</div>']}]
    elif key == 'RawBlock':
    # this is needed for the cells that have embedded video.
    # We add a special tag to those: ``` {python, .jupyter-code-cell}
    # The post-processing script then finds those and genrates separate
    # code cells that can load video.
        [format, content] = value
        if format == 'html' and 'iframe' in content:
            # Extract the video URL
            video_url = content.split('src="')[1].split('"')[0]
            # Create the Python code to display the video
            python_code = f"""
from IPython.display import display, HTML
html_code = \"""
{content}
\"""
display(HTML(html_code))
"""

            return {'t': 'CodeBlock', 'c': [['', ['python', 'jupyter-code-cell'], []], python_code]}


def process_images(key, value, format, meta):
    # Add https://pytorch.org/tutorials/ to images so that they
    # load correctly in the notebook.
    if key != 'Image':
        return None
    [ident, classes, keyvals], caption, [src, title] = value
    if not src.startswith('http'):
        while src.startswith('../'):
            src = src[3:]
        if src.startswith('/_static'):
            src = src[1:]
        src = 'https://pytorch.org/tutorials/' + src
        
    return {'t': 'Image', 'c': [[ident, classes, keyvals], caption, [src, title]]}

def process_grids(key, value, format, meta):
    # Generate side by side grid cards. Only for the two-cards layout
    # that we use in the tutorial template.
    if key == 'Div':
        [[ident, classes, keyvals], contents] = value
        if 'grid' in classes:
            columns = ['<div style="width: 45%; float: left; padding: 20px;">',
                       '<div style="width: 45%; float: right; padding: 20px;">']
            column_num = 0
            for block in contents:
                if 't' in block and block['t'] == 'Div' and 'grid-item-card' in block['c'][0][1]:
                    item_html = ''
                    for item in block['c'][1]:
                        if item['t'] == 'Para':
                            item_html += '<h2>' + ''.join(to_markdown(i) for i in item['c']) + '</h2>'
                        elif item['t'] == 'BulletList':
                            item_html += '<ul>'
                            for list_item in item['c']:
                                item_html += '<li>' + ''.join(to_markdown(i) for i in list_item[0]['c']) + '</li>'
                            item_html += '</ul>'
                    columns[column_num] += item_html
                    column_num = (column_num + 1) % 2
            columns = [column + '</div>' for column in columns]
            return {'t': 'RawBlock', 'c': ['html', ''.join(columns)]}

def is_code_block(item):
    return item['t'] == 'Code' and 'octicon' in item['c'][1]


def process_all(key, value, format, meta):
    for transform in [process_admonitions, process_images, process_grids]:
        new_value = transform(key, value, format, meta)
        if new_value is not None:
            break
    return new_value


if __name__ == "__main__":
    toJSONFilter(process_all)
