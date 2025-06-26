#!/bin/bash

if [ "$CURRENT_BRANCH" == "$BASE_BRANCH" ]; then
  echo "Running on $BASE_BRANCH branch. Skipping check."
  exit 0
fi


# Get list of deleted or renamed files in this branch compared to base
DELETED_FILES=$(git diff --name-status $BASE_BRANCH $CURRENT_BRANCH --diff-filter=DR | grep -E '\.(rst|py|md)$' | grep
| awk '{print $2}' | grep -E '\.(rst|py|md)$' | grep -v 'redirects.py')

# Check if any deleted or renamed files were found
if [ -z "$DELETED_FILES" ]; then
  echo "No deleted or renamed files found. Skipping check."
  exit 0
fi

echo "Deleted or renamed files:"
echo "$DELETED_FILES"

# Check if redirects.py has been updated
REDIRECTS_UPDATED=$(git diff --name-status $BASE_BRANCH $CURRENT_BRANCH --diff-filter=AM | grep 'redirects.py' && echo "yes" || echo "no")

if [ "$REDIRECTS_UPDATED" == "no" ]; then
  echo "ERROR: Files were deleted or renamed but redirects.py was not updated. Please update .github/scripts/redirects.py to redirect these files."
  exit 1
fi

# Check if each deleted file has a redirect entry
MISSING_REDIRECTS=0
for FILE in $DELETED_FILES; do
  # Convert file path to URL path format (remove extension and adjust path)
  REDIRECT_PATH=$(echo $FILE | sed -E 's/(.+)_source\/(.+)\.(py|rst|md|ipynb)$/\1\/\2.html/')

  # Check if this path exists in redirects.py as a key (without checking the target)
  if ! grep -q "\"$REDIRECT_PATH\":" redirects.py; then
    echo "ERROR: Missing redirect for deleted file: $FILE (should have entry for \"$REDIRECT_PATH\")"
    MISSING_REDIRECTS=1
  fi
done

if [ $MISSING_REDIRECTS -eq 1 ]; then
  echo "ERROR: Please add redirects for all deleted/renamed files to redirects.py"
  exit 1
fi

echo "All deleted/renamed files have proper redirects. Check passed!"
