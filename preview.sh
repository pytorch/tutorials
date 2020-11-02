pip install sphinx-autobuild
sphinx-autobuild --ignore "*.png" \
                 --ignore "advanced/*" \
                 --ignore "beginner/*" \
                 --ignore "intermediate/*" \
                 --ignore "prototype/*" \
                 --ignore "recipes/*" \
                 --ignore "*.zip" \
                 --ignore "*.ipynb" \
                 -D plot_gallery=0 -b html "." "_build/html"