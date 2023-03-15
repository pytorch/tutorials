set -ex

# set locale for click dependency in spacy
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Update root certificates by installing new libgnutls30
sudo apt-get update || sudo apt-get install libgnutls30
sudo apt-get update
sudo apt-get install -y --no-install-recommends unzip p7zip-full sox libsox-dev libsox-fmt-all rsync

export PATH=/opt/conda/bin:$PATH
rm -rf src
pip install -r $DIR/../requirements.txt

#Install PyTorch Nightly for test.
# Nightly - pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
# RC Link
# pip uninstall -y torch torchvision torchaudio torchtext
# pip install --pre --upgrade -f https://download.pytorch.org/whl/test/cu102/torch_test.html torch  torchvision torchaudio torchtext
# pip uninstall -y torch torchvision torchaudio torchtext
# pip install --pre --upgrade -f https://download.pytorch.org/whl/test/cu116/torch_test.html torch torchdata torchvision torchaudio torchtext

# Install two language tokenizers for Translation with TorchText tutorial
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
