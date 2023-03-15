set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Decide whether to parallelize tutorial builds, based on $JOB_BASE_NAME
export NUM_WORKERS=20

# Step 1: Generate no-plot HTML pages for all tutorials
make html-noplot
cp -r _build/html docs

# Step 2: Wait for all workers to finish
# Github should wait for all workers to finish before starting the manager.

# Step 3: Download generated with-plot HTML files and static files from S3, merge into one folder
mkdir -p docs_with_plot/docs
for ((worker_id=0;worker_id<NUM_WORKERS;worker_id++)); do
  awsv2 s3 cp s3://${BUCKET_NAME}/${COMMIT_ID}/worker_$worker_id.7z worker_$worker_id.7z
  7z x worker_$worker_id.7z -oworker_$worker_id
  yes | cp -R worker_$worker_id/docs/* docs_with_plot/docs
done

# Step 4: Copy all generated files into docs
rsync -av docs_with_plot/docs/ docs --exclude='**aws_distributed_training_tutorial*'

# Step 5: Remove INVISIBLE_CODE_BLOCK from .html/.rst.txt/.ipynb/.py files
bash $DIR/remove_invisible_code_block_batch.sh docs
python .jenkins/validate_tutorials_built.py

# Step 6: Copy generated HTML files and static files to S3
7z a manager.7z docs
awsv2 s3 cp manager.7z s3://${BUCKET_NAME}/${COMMIT_ID}/manager.7z --acl public-read

# Step 7: push new HTML files and static files to gh-pages
if [[ "$COMMIT_SOURCE" == master || "$COMMIT_SOURCE" == main ]]; then
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
