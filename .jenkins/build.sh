set -ex

sudo apt-get update
sudo apt-get install -y --no-install-recommends unzip p7zip-full sox libsox-dev libsox-fmt-all

# Install a nightly build of pytorch

# GPU, requires CUDA version 8.0
pip install cython torch_nightly -f https://download.pytorch.org/whl/nightly/cu80/torch_nightly.html

# GPU, requires CUDA version 9.0
# pip install cython torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

# GPU, requires CUDA version 9.2
# pip install cython torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html

# CPU
# pip install cython torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html


export PATH=/opt/conda/bin:$PATH
# pillow >= 4.2 will throw error when trying to write mode RGBA as JPEG,
# this is a workaround to the issue.
conda install -y sphinx==1.8.2 pandas pillow=4.1.1
# PyTorch Theme
rm -rf src
pip install -e git+git://github.com/pytorch/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
pip install sphinx-gallery tqdm matplotlib ipython scikit-image

git clone https://github.com/pytorch/vision --quiet
pushd vision
pip install . --no-deps  # We don't want it to install the stock PyTorch version from pip
popd

git clone https://github.com/pytorch/audio --quiet
pushd audio
python setup.py install
popd

# We will fix the hybrid frontend tutorials when the API is stable
rm beginner_source/hybrid_frontend/learning_hybrid_frontend_through_example_tutorial.py || true
rm beginner_source/hybrid_frontend/introduction_to_hybrid_frontend_tutorial.py || true

make docs

rm -rf vision
rm -rf audio
