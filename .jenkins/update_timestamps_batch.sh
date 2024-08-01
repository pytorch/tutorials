#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Check if SPHINX_SHOULD_RUN is already set
if [ -z "$SPHINX_SHOULD_RUN" ]; then
    # If not set, retrieve it using get_sphinx_filenames.py and export it
    SPHINX_SHOULD_RUN=$(python "$DIR/get_sphinx_filenames.py")
    export SPHINX_SHOULD_RUN
fi

# Convert the pipe-separated filenames into an array
IFS='|' read -r -a file_array <<< "$SPHINX_SHOULD_RUN"

# Loop through each file and update timestamps if it exists
for file in "${file_array[@]}"; do
    file="$DIR/../$file"
    if [ -f "$file" ]; then
        python "$DIR/update_timestamps.py" "$file"
    fi
done
