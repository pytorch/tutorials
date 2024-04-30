#!/bin/bash

set -ex

export BUCKET_NAME=pytorch-tutorial-build-pull-request

# set locale for click dependency in spacy
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# Update root certificates by installing new libgnutls30

# Install pandoc (does not install from pypi)
sudo apt-get update
sudo apt-get install -y pandoc

# NS: Path to python runtime should already be part of docker container
# export PATH=/opt/conda/bin:$PATH

#Install PyTorch Nightly for test.
# Nightly - pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
# Install 2.2 for testing - uncomment to install nightly binaries (update the version as needed).
# pip uninstall -y torch torchvision torchaudio torchtext torchdata
# pip3 install torch==2.3.0 torchvision torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/test/cu121 

# Install two language tokenizers for Translation with TorchText tutorial
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

awsv2 -i
awsv2 configure set default.s3.multipart_threshold 5120MB

# Decide whether to parallelize tutorial builds, based on $JOB_BASE_NAME
if [[ "${JOB_TYPE}" == "worker" ]]; then
  # Step 1: Remove runnable code from tutorials that are not supposed to be run
  python $DIR/remove_runnable_code.py beginner_source/aws_distributed_training_tutorial.py beginner_source/aws_distributed_training_tutorial.py || true
  # python $DIR/remove_runnable_code.py advanced_source/ddp_pipeline_tutorial.py advanced_source/ddp_pipeline_tutorial.py || true
  # Temp remove for mnist download issue. (Re-enabled for 1.8.1)
  # python $DIR/remove_runnable_code.py beginner_source/fgsm_tutorial.py beginner_source/fgsm_tutorial.py || true
  # python $DIR/remove_runnable_code.py intermediate_source/spatial_transformer_tutorial.py intermediate_source/spatial_transformer_tutorial.py || true
  # Temp remove for 1.10 release.
  # python $DIR/remove_runnable_code.py advanced_source/neural_style_tutorial.py advanced_source/neural_style_tutorial.py || true

  # TODO: Fix bugs in these tutorials to make them runnable again
  # python $DIR/remove_runnable_code.py beginner_source/audio_classifier_tutorial.py beginner_source/audio_classifier_tutorial.py || true

  # Remove runnable code from tensorboard_profiler_tutorial.py as it frequently crashes, see https://github.com/pytorch/pytorch/issues/74139
  # python $DIR/remove_runnable_code.py intermediate_source/tensorboard_profiler_tutorial.py intermediate_source/tensorboard_profiler_tutorial.py || true

  # Step 2: Keep certain tutorials based on file count, and remove runnable code in all other tutorials
  # IMPORTANT NOTE: We assume that each tutorial has a UNIQUE filename.
  FILES_TO_RUN=$(python .jenkins/get_files_to_run.py)
  echo "FILES_TO_RUN: " ${FILES_TO_RUN}
  # Files to run must be accessible to subprocessed (at least to `download_data.py`)
  export FILES_TO_RUN

  # Step 3: Run `make docs` to generate HTML files and static files for these tutorials
  make docs

  # Step 3.1: Run the post-processing script:
  python .jenkins/post_process_notebooks.py

  # Step 4: If any of the generated files are not related the tutorial files we want to run,
  # then we remove them
  set +x
  for filename in $(find docs/beginner docs/intermediate docs/advanced docs/recipes docs/prototype -name '*.html'); do
    file_basename=$(basename $filename .html)
    if [[ ! " ${FILES_TO_RUN} " =~ " ${file_basename} " ]]; then
      echo "removing $filename"
      rm $filename
    fi
  done
  for filename in $(find docs/beginner docs/intermediate docs/advanced docs/recipes docs/prototype -name '*.rst'); do
    file_basename=$(basename $filename .rst)
    if [[ ! " ${FILES_TO_RUN} " =~ " ${file_basename} " ]]; then
      echo "removing $filename"
      rm $filename
    fi
  done
  for filename in $(find docs/_downloads -name '*.py'); do
    file_basename=$(basename $filename .py)
    if [[ ! " ${FILES_TO_RUN} " =~ " ${file_basename} " ]]; then
      echo "removing $filename"
      rm $filename
    fi
  done
  for filename in $(find docs/_downloads -name '*.ipynb'); do
    file_basename=$(basename $filename .ipynb)
    if [[ ! " ${FILES_TO_RUN} " =~ " ${file_basename} " ]]; then
      echo "removing $filename"
      rm $filename
    fi
  done
  for filename in $(find docs/_sources/beginner docs/_sources/intermediate docs/_sources/advanced docs/_sources/recipes -name '*.rst.txt'); do
    file_basename=$(basename $filename .rst.txt)
    if [[ ! " ${FILES_TO_RUN} " =~ " ${file_basename} " ]]; then
      echo "removing $filename"
      rm $filename
    fi
  done
  for filename in $(find docs/.doctrees/beginner docs/.doctrees/intermediate docs/.doctrees/advanced docs/.doctrees/recipes docs/.doctrees/prototype -name '*.doctree'); do
    file_basename=$(basename $filename .doctree)
    if [[ ! " ${FILES_TO_RUN} " =~ " ${file_basename} " ]]; then
      echo "removing $filename"
      rm $filename
    fi
  done
  set -x

  # Step 5: Remove INVISIBLE_CODE_BLOCK from .html/.rst.txt/.ipynb/.py files
  bash $DIR/remove_invisible_code_block_batch.sh docs
  python .jenkins/validate_tutorials_built.py

  # Step 6: Copy generated files to S3, tag with commit ID
  7z a worker_${WORKER_ID}.7z docs
  awsv2 s3 cp worker_${WORKER_ID}.7z s3://${BUCKET_NAME}/${COMMIT_ID}/worker_${WORKER_ID}.7z
elif [[ "${JOB_TYPE}" == "manager" ]]; then
  # Step 1: Generate no-plot HTML pages for all tutorials
  make html-noplot
  cp -r _build/html docs

  # Step 2: Wait for all workers to finish
  # Don't actually need to do this because gha will wait

  # Step 3: Download generated with-plot HTML files and static files from S3, merge into one folder
  mkdir -p docs_with_plot/docs
  for ((worker_id=1;worker_id<NUM_WORKERS+1;worker_id++)); do
    awsv2 s3 cp s3://${BUCKET_NAME}/${COMMIT_ID}/worker_$worker_id.7z worker_$worker_id.7z
    7z x worker_$worker_id.7z -oworker_$worker_id
    yes | cp -R worker_$worker_id/docs/* docs_with_plot/docs
  done

  # Step 4: Copy all generated files into docs
  rsync -av docs_with_plot/docs/ docs --exclude='**aws_distributed_training_tutorial*'

  # Step 5: Remove INVISIBLE_CODE_BLOCK from .html/.rst.txt/.ipynb/.py files
  bash $DIR/remove_invisible_code_block_batch.sh docs
  python .jenkins/validate_tutorials_built.py

  # Step 5.1: Run post-processing script on .ipynb files:
  python .jenkins/post_process_notebooks.py

  # Step 6: Copy generated HTML files and static files to S3
  7z a manager.7z docs
  awsv2 s3 cp manager.7z s3://${BUCKET_NAME}/${COMMIT_ID}/manager.7z

  # Step 7: push new HTML files and static files to gh-pages
  if [[ "$COMMIT_SOURCE" == "refs/heads/master" || "$COMMIT_SOURCE" == "refs/heads/main" ]]; then
    git clone https://github.com/pytorch/tutorials.git -b gh-pages gh-pages
    cp -r docs/* gh-pages/
    pushd gh-pages
    # DANGER! DO NOT REMOVE THE `set +x` SETTING HERE!
    set +x
    git remote set-url origin https://pytorchbot:${GITHUB_PYTORCHBOT_TOKEN}@github.com/pytorch/tutorials.git
    set -x
    git add -f -A || true
    git config user.email "soumith+bot@pytorch.org"
    git config user.name "pytorchbot"
    git commit -m "Automated tutorials push" || true
    git status
    git push origin gh-pages
  fi
else
  make docs
fi
