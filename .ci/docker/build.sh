#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

IMAGE_NAME="$1"
shift

export UBUNTU_VERSION="22.04"
export CUDA_VERSION="12.6.3"

export BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
echo "Building ${IMAGE_NAME} Docker image"

# On OSDC the image is built by the out-of-cluster BuildKit pool through a
# remote buildx builder. That builder has nowhere to load an image into, so the
# result has to go straight to the registry.
if [[ -n "${REMOTE_BUILDKIT:-}" ]]; then
  BUILD_CMD=(docker buildx build --push)
else
  BUILD_CMD=(docker build)
fi

"${BUILD_CMD[@]}" \
  --no-cache \
  --progress=plain \
  -f Dockerfile \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  "$@" \
  .
