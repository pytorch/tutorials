#!/bin/bash

SOURCEDIR=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

directories=("$SOURCEDIR/beginner_source" "$SOURCEDIR/intermediate_source" "$SOURCEDIR/advanced_source")

for dir in "${directories[@]}"; do
    # Process .py and .rst files in the current directory
    for file in "$dir"/*.{py,rst}; do
        if [ -f "$file" ]; then
            python "$DIR/update_timestamps.py" "$file"
        fi
    done
done