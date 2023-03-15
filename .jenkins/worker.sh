set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Decide whether to parallelize tutorial builds, based on $JOB_BASE_NAME
export NUM_WORKERS=20

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
export WORKER_ID=$(echo "${JOB_BASE_NAME}" | tr -dc '0-9')
FILES_TO_RUN=$(python .jenkins/get_files_to_run.py)
echo "FILES_TO_RUN: " ${FILES_TO_RUN}

# Step 3: Run `make docs` to generate HTML files and static files for these tutorials
make docs

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
