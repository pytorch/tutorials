#!/bin/bash

set -ex

rm -rf build
rm -rf custom_class_project/build

pushd custom_class_project
  mkdir build
  (cd build && cmake CXXFLAGS="-DNO_PICKLE" -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..)
  (cd build && make)
  python custom_test.py
  python save.py
  ! python export_attr.py
popd

mkdir build
(cd build && cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..)
(cd build && make)
mv custom_class_project/foo.pt build/foo.pt
(cd build && ./infer)
