from pandocfilters import toJSONFilter, Div, RawBlock, Para, Str, Space, Link, Code, CodeBlock
import markdown
import re

def to_markdown(item):
    if item['t'] == 'Str':
        return item['c']
    elif item['t'] == 'Space':
        return ' '
    elif item['t'] == 'Link':
        # Assuming the link text is always in the first item
        return f"[{item['c'][1][0]['c']}]({item['c'][2][0]})"
    elif item['t'] == 'Code':
        return f"`{item['c'][1]}`"
    elif item['t'] == 'CodeBlock':
        return f"```\n{item['c'][1]}\n```"

def process_admonitions(key, value, format, meta):
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
            if 't' in block and block['t'] == 'Para':
                for item in block['c']:
                    if item['t'] == 'Str':
                        note_content.append(Str(item['c']))
                    elif item['t'] == 'Space':
                        note_content.append(Space())
                    elif item['t'] == 'Link':
                        note_content.append(Link(*item['c']))
                    elif item['t'] == 'Code':
                        note_content.append(Code(*item['c']))
            elif 't' in block and block['t'] == 'CodeBlock':
                note_content.append(CodeBlock(*block['c']))

        note_content_md = ''.join(to_markdown(item) for item in note_content)
        html_content = markdown.markdown(note_content_md)

        return [{'t': 'RawBlock', 'c': ['html', f'<div style="background-color: {color}; color: #fff; font-weight: 700; padding-left: 10px; padding-top: 5px; padding-bottom: 5px">{label}</div>']}, {'t': 'RawBlock', 'c': ['html', '<div style="background-color: #f3f4f7; padding-left: 10px; padding-top: 10px; padding-bottom: 10px; padding-right: 10px">']}, {'t': 'RawBlock', 'c': ['html', html_content]}, {'t': 'RawBlock', 'c': ['html', '</div>']}]

    elif key == 'RawBlock':
        [format, content] = value
        if format == 'html' and 'iframe' in content:
            # Extract the video URL
            video_url = content.split('src="')[1].split('"')[0]
            # Create the Python code to display the video
            html_code = f"""
from IPython.display import display, HTML
html_code = \"""
{content}
\"""
display(HTML(html_code))
"""

if __name__ == "__main__":
    toJSONFilter(process_admonitions)
