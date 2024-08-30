#!/bin/bash

# Work around bug where devtoolset replaces sudo and breaks it.
as_ci_user() {
  # NB: unsetting the environment variables works around a conda bug
  # https://github.com/conda/conda/issues/6576
  # NB: Pass on PATH and LD_LIBRARY_PATH to sudo invocation
  # NB: This must be run from a directory that the user has access to,
  # works around https://github.com/conda/conda-package-handling/pull/34
  sudo -E -H -u ci-user env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" $*
}

conda_install() {
  # Ensure that the install command don't upgrade/downgrade Python
  # This should be called as
  #   conda_install pkg1 pkg2 ... [-c channel]
  as_ci_user conda install -q -n py_$ANACONDA_PYTHON_VERSION -y python="$ANACONDA_PYTHON_VERSION" $*
}

conda_run() {
  as_ci_user conda run -n py_$ANACONDA_PYTHON_VERSION --no-capture-output $*
}

pip_install() {
  as_ci_user conda run -n py_$ANACONDA_PYTHON_VERSION pip3 install --progress-bar off $*
}
