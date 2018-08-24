sudo apt-get update
sudo apt-get install -y --no-install-recommends unzip p7zip-full sox libsox-dev libsox-fmt-all

export PATH=/opt/conda/bin:$PATH

## Build pytorch
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic pytorch dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake=3.5.0 cffi typing
conda install -c mingfeima mkldnn

# Add LAPACK support for the GPU
conda install -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9

# Clone pytorch repo and build from scratch
git clone https://github.com/pytorch/pytorch.git
pushd pytorch
git submodule update --init
.jenkins/pytorch/build.sh
popd

## install doc dependencies

# pillow >= 4.2 will throw error when trying to write mode RGBA as JPEG,
# this is a workaround to the issue.
conda install -y sphinx pandas pillow=4.1.1
pip install sphinx-gallery sphinx_rtd_theme tqdm matplotlib ipython

git clone https://github.com/pytorch/vision --quiet
pushd vision
pip install . --no-deps  # We don't want it to install the stock PyTorch version from pip
popd

git clone https://github.com/pytorch/audio --quiet
pushd audio
python setup.py install
popd

# Download dataset for beginner_source/dcgan_faces_tutorial.py
curl https://s3.amazonaws.com/pytorch-tutorial-assets/img_align_celeba.zip --output img_align_celeba.zip
sudo mkdir -p /home/ubuntu/facebook/datasets/celeba
sudo chmod -R 0777 /home/ubuntu/facebook/datasets/celeba
unzip img_align_celeba.zip -d /home/ubuntu/facebook/datasets/celeba > null

# Download dataset for beginner_source/hybrid_frontend/introduction_to_hybrid_frontend_tutorial.py
mkdir data/
curl https://s3.amazonaws.com/pytorch-tutorial-assets/iris.data --output data/iris.data

# Download dataset for beginner_source/chatbot_tutorial.py
curl https://s3.amazonaws.com/pytorch-tutorial-assets/cornell_movie_dialogs_corpus.zip --output cornell_movie_dialogs_corpus.zip
mkdir -p beginner_source/data
unzip cornell_movie_dialogs_corpus.zip -d beginner_source/data/ > null

# Download dataset for beginner_source/audio_classifier_tutorial.py
curl https://s3.amazonaws.com/pytorch-tutorial-assets/UrbanSound8K.tar.gz --output UrbanSound8K.tar.gz
tar -xzf UrbanSound8K.tar.gz -C ./beginner_source

# Download model for beginner_source/fgsm_tutorial.py
curl https://s3.amazonaws.com/pytorch-tutorial-assets/lenet_mnist_model.pth --output ./beginner_source/lenet_mnist_model.pth

# We will fix the hybrid frontend tutorials when the API is stable
rm beginner_source/hybrid_frontend/learning_hybrid_frontend_through_example_tutorial.py
rm beginner_source/hybrid_frontend/introduction_to_hybrid_frontend_tutorial.py

make docs

rm -rf vision
rm -rf audio
