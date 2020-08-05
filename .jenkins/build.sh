set -ex

if [[ "$COMMIT_SOURCE" == master ]]; then
  export BUCKET_NAME=pytorch-tutorial-build-master
else
  export BUCKET_NAME=pytorch-tutorial-build-pull-request
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

sudo apt-get update
sudo apt-get install -y --no-install-recommends unzip p7zip-full sox libsox-dev libsox-fmt-all rsync

export PATH=/opt/conda/bin:$PATH
rm -rf src
pip install -r $DIR/../requirements.txt

# export PATH=/opt/conda/bin:$PATH
# pip install sphinx==1.8.2 pandas

# For Tensorboard. Until 1.14 moves to the release channel.
pip install tb-nightly

# Install two language tokenizers for Translation with TorchText tutorial
python -m spacy download en
python -m spacy download de

# PyTorch Theme
rm -rf src
pip install -e git+git://github.com/pytorch/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
# pillow >= 4.2 will throw error when trying to write mode RGBA as JPEG,
# this is a workaround to the issue.
pip install sphinx-gallery==0.3.1 tqdm matplotlib ipython pillow==4.1.1

aws configure set default.s3.multipart_threshold 5120MB

# Decide whether to parallelize tutorial builds, based on $JOB_BASE_NAME
export NUM_WORKERS=20
if [[ "${JOB_BASE_NAME}" == *worker_* ]]; then
  # Step 1: Remove runnable code from tutorials that are not supposed to be run
  python $DIR/remove_runnable_code.py beginner_source/aws_distributed_training_tutorial.py beginner_source/aws_distributed_training_tutorial.py || true
  # TODO: Fix bugs in these tutorials to make them runnable again
  # python $DIR/remove_runnable_code.py beginner_source/audio_classifier_tutorial.py beginner_source/audio_classifier_tutorial.py || true

  # Step 2: Keep certain tutorials based on file count, and remove runnable code in all other tutorials
  # IMPORTANT NOTE: We assume that each tutorial has a UNIQUE filename.
  export WORKER_ID=$(echo "${JOB_BASE_NAME}" | tr -dc '0-9')
  count=0
  FILES_TO_RUN=()
  for filename in $(find beginner_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename $filename
    else
      echo "Keeping "$filename
      FILES_TO_RUN+=($(basename $filename .py))
    fi
    count=$((count+1))
  done
  for filename in $(find intermediate_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename $filename
    else
      echo "Keeping "$filename
      FILES_TO_RUN+=($(basename $filename .py))
    fi
    count=$((count+1))
  done
  for filename in $(find advanced_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename $filename
    else
      echo "Keeping "$filename
      FILES_TO_RUN+=($(basename $filename .py))
    fi
    count=$((count+1))
   done
   for filename in $(find recipes_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename $filename
    else
      echo "Keeping "$filename
      FILES_TO_RUN+=($(basename $filename .py))
    fi
    count=$((count+1))
   done
   for filename in $(find prototype_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename $filename
    else
      echo "Keeping "$filename
      FILES_TO_RUN+=($(basename $filename .py))
    fi
    count=$((count+1))
  done
  echo "FILES_TO_RUN: " ${FILES_TO_RUN[@]}

  # Step 3: Run `make docs` to generate HTML files and static files for these tutorials
  make docs

  # Step 4: If any of the generated files are not related the tutorial files we want to run,
  # then we remove them
  for filename in $(find docs/beginner docs/intermediate docs/advanced docs/recipes docs/prototype -name '*.html'); do
    file_basename=$(basename $filename .html)
    if [[ ! " ${FILES_TO_RUN[@]} " =~ " ${file_basename} " ]]; then
      rm $filename
    fi
  done
  for filename in $(find docs/beginner docs/intermediate docs/advanced docs/recipes docs/prototype -name '*.rst'); do
    file_basename=$(basename $filename .rst)
    if [[ ! " ${FILES_TO_RUN[@]} " =~ " ${file_basename} " ]]; then
      rm $filename
    fi
  done
  for filename in $(find docs/_downloads -name '*.py'); do
    file_basename=$(basename $filename .py)
    if [[ ! " ${FILES_TO_RUN[@]} " =~ " ${file_basename} " ]]; then
      rm $filename
    fi
  done
  for filename in $(find docs/_downloads -name '*.ipynb'); do
    file_basename=$(basename $filename .ipynb)
    if [[ ! " ${FILES_TO_RUN[@]} " =~ " ${file_basename} " ]]; then
      rm $filename
    fi
  done
  for filename in $(find docs/_sources/beginner docs/_sources/intermediate docs/_sources/advanced docs/_sources/recipes -name '*.rst.txt'); do
    file_basename=$(basename $filename .rst.txt)
    if [[ ! " ${FILES_TO_RUN[@]} " =~ " ${file_basename} " ]]; then
      rm $filename
    fi
  done
  for filename in $(find docs/.doctrees/beginner docs/.doctrees/intermediate docs/.doctrees/advanced docs/.doctrees/recipes docs/.doctrees/prototype -name '*.doctree'); do
    file_basename=$(basename $filename .doctree)
    if [[ ! " ${FILES_TO_RUN[@]} " =~ " ${file_basename} " ]]; then
      rm $filename
    fi
  done

  # Step 5: Remove INVISIBLE_CODE_BLOCK from .html/.rst.txt/.ipynb/.py files
  bash $DIR/remove_invisible_code_block_batch.sh docs

  # Step 6: Copy generated files to S3, tag with commit ID
  7z a worker_${WORKER_ID}.7z docs
  aws s3 cp worker_${WORKER_ID}.7z s3://${BUCKET_NAME}/${COMMIT_ID}/worker_${WORKER_ID}.7z --acl public-read
elif [[ "${JOB_BASE_NAME}" == *manager ]]; then
  # Step 1: Generate no-plot HTML pages for all tutorials
  make html-noplot
  cp -r _build/html docs

  # Step 2: Wait for all workers to finish
  set +e
  for ((worker_id=0;worker_id<NUM_WORKERS;worker_id++)); do
    until aws s3api head-object --bucket ${BUCKET_NAME} --key ${COMMIT_ID}/worker_$worker_id.7z
    do
      echo "Waiting for worker $worker_id to finish..."
      sleep 5
    done
  done
  set -e

  # Step 3: Download generated with-plot HTML files and static files from S3, merge into one folder
  mkdir -p docs_with_plot/docs
  for ((worker_id=0;worker_id<NUM_WORKERS;worker_id++)); do
    aws s3 cp s3://${BUCKET_NAME}/${COMMIT_ID}/worker_$worker_id.7z worker_$worker_id.7z
    7z x worker_$worker_id.7z -oworker_$worker_id
    yes | cp -R worker_$worker_id/docs/* docs_with_plot/docs
  done

  # Step 4: Copy all generated files into docs
  rsync -av docs_with_plot/docs/ docs --exclude='**aws_distributed_training_tutorial*'

  # Step 5: Remove INVISIBLE_CODE_BLOCK from .html/.rst.txt/.ipynb/.py files
  bash $DIR/remove_invisible_code_block_batch.sh docs

  # Step 6: Copy generated HTML files and static files to S3
  7z a manager.7z docs
  aws s3 cp manager.7z s3://${BUCKET_NAME}/${COMMIT_ID}/manager.7z --acl public-read

  # Step 7: push new HTML files and static files to gh-pages
  if [[ "$COMMIT_SOURCE" == master ]]; then
    git clone https://github.com/pytorch/tutorials.git -b gh-pages gh-pages
    cp -r docs/* gh-pages/
    pushd gh-pages
    # DANGER! DO NOT REMOVE THE `set +x` SETTING HERE!
    set +x
    git remote set-url origin https://${GITHUB_PYTORCHBOT_USERNAME}:${GITHUB_PYTORCHBOT_TOKEN}@github.com/pytorch/tutorials.git
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
