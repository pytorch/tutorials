set -ex

if [[ "$COMMIT_SOURCE" == master ]]; then
  export BUCKET_NAME=pytorch-tutorial-build-master
else
  export BUCKET_NAME=pytorch-tutorial-build-pull-request
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

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
conda install -y sphinx==1.8.2 pandas
# PyTorch Theme
rm -rf src
pip install -e git+git://github.com/pytorch/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
# pillow >= 4.2 will throw error when trying to write mode RGBA as JPEG,
# this is a workaround to the issue.
pip install sphinx-gallery tqdm matplotlib ipython pillow==4.1.1

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

# Decide whether to parallelize tutorial builds, based on $JOB_BASE_NAME
export NUM_WORKERS=20
if [[ "${JOB_BASE_NAME}" == *worker_* ]]; then
  # Step 1: Keep certain tutorials based on file count, and remove runnable code in all other tutorials
  export WORKER_ID=$(echo "${JOB_BASE_NAME}" | tr -dc '0-9')
  count=0
  for filename in $(find beginner_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename
    else
      echo "Keeping "$filename
    fi
    count=$((count+1))
  done
  for filename in $(find intermediate_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename
    else
      echo "Keeping "$filename
    fi
    count=$((count+1))
  done
  for filename in $(find advanced_source/ -name '*.py' -not -path '*/data/*'); do
    if [ $(($count % $NUM_WORKERS)) != $WORKER_ID ]; then
      echo "Removing runnable code from "$filename
      python $DIR/remove_runnable_code.py $filename
    else
      echo "Keeping "$filename
    fi
    count=$((count+1))
  done

  # Step 2: Run `make docs` to generate HTML files and static files for these tutorials
  make docs

  # Step 3: Enable all tutorial Python files again
  git checkout -- beginner_source/
  git checkout -- intermediate_source/
  git checkout -- advanced_source/

  # Step 4: Remove all HTML files that don't contain runnable code
  for filename in $(find docs/beginner/ -name '*.html'); do
    if grep -Fxq "%%%%%%RUNNABLE_CODE_REMOVED%%%%%%" $filename then
      echo "Removing " $filename
      rm $filename
    fi
  done
  for filename in $(find docs/intermediate/ -name '*.html'); do
    if grep -Fxq "%%%%%%RUNNABLE_CODE_REMOVED%%%%%%" $filename then
      echo "Removing " $filename
      rm $filename
    fi
  done
  for filename in $(find docs/advanced/ -name '*.html'); do
    if grep -Fxq "%%%%%%RUNNABLE_CODE_REMOVED%%%%%%" $filename then
      echo "Removing " $filename
      rm $filename
    fi
  done

  # Step 5: Copy generated HTML files and static files to S3, tag with commit ID
  7z a worker_${WORKER_ID}.7z docs
  aws s3 cp worker_${WORKER_ID}.7z s3://${BUCKET_NAME}/${COMMIT_ID}/worker_${WORKER_ID}.7z
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
  mkdir docs_with_plot/
  for ((worker_id=0;worker_id<NUM_WORKERS;worker_id++)); do
    aws s3 cp s3://${BUCKET_NAME}/${COMMIT_ID}/worker_$worker_id.7z worker_$worker_id.7z
    7z x worker_$worker_id.7z -oworker_$worker_id
    yes | cp -rf worker_$worker_id/docs docs_with_plot/docs
  done

  # Step 4: Copy plots into the no-plot HTML pages
  for filename in $(find docs/beginner/ -name '*.html'); do
    python $DIR/replace_tutorial_html_content.py $filename docs_with_plot/$filename $filename
  done
  for filename in $(find docs/intermediate/ -name '*.html'); do
    python $DIR/replace_tutorial_html_content.py $filename docs_with_plot/$filename $filename
  done
  for filename in $(find docs/advanced/ -name '*.html'); do
    python $DIR/replace_tutorial_html_content.py $filename docs_with_plot/$filename $filename
  done

  # Step 5: Copy all static files into docs
  rsync -av docs_with_plot/docs/ docs --exclude beginner --exclude intermediate --exclude advanced
else
  make docs
fi

rm -rf vision
rm -rf audio
