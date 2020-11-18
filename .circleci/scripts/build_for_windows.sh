#!/bin/bash
set -eux -o pipefail

retry () {
  $*  || (sleep 1 && $*) || (sleep 2 && $*)
}

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PROJECT_DIR="${SOURCE_DIR}/../.."
pushd $SOURCE_DIR

#install wget and make
curl -k https://ymu.dl.osdn.jp/mingw/68260/mingw-get-0.6.3-mingw32-pre-20170905-1-bin.zip -o mingw32.zip
unzip mingw32.zip -d mingw32
mingw32/bin/mingw-get.exe install mingw32-make
mingw32/bin/mingw-get.exe install msys-findutils
mv mingw32/bin/mingw32-make.exe mingw32/bin/make.exe
curl -k https://eternallybored.org/misc/wget/1.20.3/64/wget.exe -o mingw32/bin/wget.exe
export PATH="${SOURCE_DIR}/mingw32/bin:${SOURCE_DIR}/mingw32/msys/1.0/bin:$PATH"

#install anaconda3
export CONDA_HOME="${SOURCE_DIR}/conda"
export tmp_conda="${SOURCE_DIR}/conda"
export miniconda_exe="${SOURCE_DIR}/miniconda.exe"
rm -rf conda miniconda.exe
curl -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
./install_conda.bat
export PATH="${tmp_conda}:${tmp_conda}/Library/usr/bin:${tmp_conda}/Library/bin:${tmp_conda}/Scripts:${tmp_conda}/bin:$PATH"

eval "$(conda shell.bash hook)"
conda create -qyn testenv python=3.7
conda activate testenv

conda install sphinx
pip install sphinx_gallery==0.3.1 sphinx-copybutton flask pandas spacy ipython scipy pySoundFile scikit-image
pip install -e git+git://github.com/pytorch/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
pip install ray[tune] tabulate
conda install -yq -c pytorch "cudatoolkit=10.1" pytorch torchvision torchtext
conda install torchaudio -c pytorch-test
python -m spacy download de
python -m spacy download en
pushd ${PROJECT_DIR}
DIR=.jenkins
export NUM_WORKERS=4

if [[ "${CIRCLE_JOB}" == *worker_* ]]; then
  python $DIR/remove_runnable_code.py intermediate_source/model_parallel_tutorial.py intermediate_source/model_parallel_tutorial.py || true
  python $DIR/remove_runnable_code.py advanced_source/static_quantization_tutorial.py advanced_source/static_quantization_tutorial.py || true
  python $DIR/remove_runnable_code.py beginner_source/hyperparameter_tuning_tutorial.py beginner_source/hyperparameter_tuning_tutorial.py || true

  export WORKER_ID=$(echo "${CIRCLE_JOB}" | tr -dc '0-9')
  count=0
  FILES_TO_RUN=()
  for work_directory  in beginner_source intermediate_source advanced_source recipes_source prototype_source; do
    for filename in $(find $work_directory -name '\*.py' -not -path '\*/data/\*'); do
      if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
        echo "Removing runnable code from "$filename
        python $DIR/remove_runnable_code.py $filename $filename
      else
        echo "Keeping "$filename
        FILES_TO_RUN+=($(basename $filename .py))
      fi
      count=$((count+1))
    done
  done
  echo "FILES_TO_RUN: " ${FILES_TO_RUN[@]}
fi

if [[ ! -d advanced_source/data || ! -d beginner_source/data || ! -d intermediate_source/data || ! -d prototype_source/data ]];then
    make download
fi

make html
