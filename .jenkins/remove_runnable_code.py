import sys

STATE_IN_MULTILINE_COMMENT_BLOCK_DOUBLE_QUOTE = "STATE_IN_MULTILINE_COMMENT_BLOCK_DOUBLE_QUOTE"
STATE_IN_MULTILINE_COMMENT_BLOCK_SINGLE_QUOTE = "STATE_IN_MULTILINE_COMMENT_BLOCK_SINGLE_QUOTE"
STATE_NORMAL = "STATE_NORMAL"

python_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(python_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    ret_lines = []
    state = STATE_NORMAL
    for line in lines:
        if state == STATE_NORMAL:
            if line.startswith('#'):
                ret_lines.append(line)
                state = STATE_NORMAL
            elif ((line.startswith('"""') or line.startswith('r"""')) and
                    line.endswith('"""')):
                ret_lines.append(line)
                state = STATE_NORMAL
            elif line.startswith('"""') or line.startswith('r"""'):
                ret_lines.append(line)
                state = STATE_IN_MULTILINE_COMMENT_BLOCK_DOUBLE_QUOTE
            elif ((line.startswith("'''") or line.startswith("r'''")) and
                    line.endswith("'''")):
                ret_lines.append(line)
                state = STATE_NORMAL
            elif line.startswith("'''") or line.startswith("r'''"):
                ret_lines.append(line)
                state = STATE_IN_MULTILINE_COMMENT_BLOCK_SINGLE_QUOTE
            else:
                ret_lines.append("\n")
                state = STATE_NORMAL
        elif state == STATE_IN_MULTILINE_COMMENT_BLOCK_DOUBLE_QUOTE:
            if line.startswith('"""'):
                ret_lines.append(line)
                state = STATE_NORMAL
            else:
                ret_lines.append(line)
                state = STATE_IN_MULTILINE_COMMENT_BLOCK_DOUBLE_QUOTE
        elif state == STATE_IN_MULTILINE_COMMENT_BLOCK_SINGLE_QUOTE:
            if line.startswith("'''"):
                ret_lines.append(line)
                state = STATE_NORMAL
            else:
                ret_lines.append(line)
                state = STATE_IN_MULTILINE_COMMENT_BLOCK_SINGLE_QUOTE

ret_lines.append("\n# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%")

with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in ret_lines:
        file.write(line)
