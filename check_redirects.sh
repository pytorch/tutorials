#!/bin/bash

# Get the base branch (usually main or master)
BASE_BRANCH="main"
CURRENT_BRANCH=$(git branch --show-current)

# Get list of deleted or renamed files in this branch compared to base
DELETED_FILES=$(git diff --name-status $BASE_BRANCH $CURRENT_BRANCH | grep '^D\|^R' | awk '{print $2}' | grep -E '\.(rst|py|ipynb)$' | grep -v 'redirects.py')

if [ -z "$DELETED_FILES" ]; then
  echo "No deleted or renamed files found. Skipping check."
  exit 0
fi

echo "Deleted or renamed files:"
echo "$DELETED_FILES"

# Check if redirects.py has been updated
REDIRECTS_UPDATED=$(git diff --name-status $BASE_BRANCH $CURRENT_BRANCH | grep -E '^M|^A' | grep 'redirects.py' && echo "yes" || echo "no")

if [ "$REDIRECTS_UPDATED" == "no" ]; then
  echo "ERROR: Files were deleted or renamed but redirects.py was not updated."
  exit 1
fi

# Check if each deleted file has a redirect entry
MISSING_REDIRECTS=0
for FILE in $DELETED_FILES; do
  # Convert file path to URL path format (remove extension and adjust path)
  URL_PATH=$(echo $FILE | sed 's/\.rst$//g' | sed 's/\.py$//g' | sed 's/\.ipynb$//g' | sed 's/^tutorials\///g')
  
  # Check if this path exists in redirects.py
  if ! grep -q "\"$URL_PATH\"" tutorials/redirects.py; then
    echo "ERROR: Missing redirect for deleted file: $FILE (URL path: $URL_PATH)"
    MISSING_REDIRECTS=1
  fi
done

if [ $MISSING_REDIRECTS -eq 1 ]; then
  echo "ERROR: Please add redirects for all deleted/renamed files to redirects.py"
  exit 1
fi

echo "All deleted/renamed files have proper redirects. Check passed!"

