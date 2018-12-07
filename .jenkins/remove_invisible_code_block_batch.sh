BUILDDIR=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Remove INVISIBLE_CODE_BLOCK from .html/.rst/.rst.txt/.ipynb/.py files
for filename in $(find $BUILDDIR/beginner $BUILDDIR/intermediate $BUILDDIR/advanced -name '*.html'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_html.py $filename $filename
done
for filename in $(find $BUILDDIR/_sources/beginner $BUILDDIR/_sources/intermediate $BUILDDIR/_sources/advanced -name '*.rst.txt'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_rst_txt.py $filename $filename
done
for filename in $(find $BUILDDIR/_downloads -name '*.ipynb'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_ipynb.py $filename $filename
done
for filename in $(find $BUILDDIR/_downloads -name '*.py'); do
    echo "Removing INVISIBLE_CODE_BLOCK from " $filename
    python $DIR/remove_invisible_code_block_from_py.py $filename $filename
done
