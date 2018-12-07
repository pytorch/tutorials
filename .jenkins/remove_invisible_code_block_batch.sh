BUILDDIR=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Remove INVISIBLE_CODE_BLOCK from all HTML files
for filename in $(find $BUILDDIR/beginner -name '*.html'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_html.py $filename $filename
done
for filename in $(find $BUILDDIR/intermediate -name '*.html'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_html.py $filename $filename
done
for filename in $(find $BUILDDIR/advanced -name '*.html'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_html.py $filename $filename
done