#!/bin/bash

set -ex

rm -rf build
rm -rf custom_class_project/build

pushd custom_class_project
  mkdir build
  (cd build && cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..)
  (cd build && make)
  python export_attr.py
popd
