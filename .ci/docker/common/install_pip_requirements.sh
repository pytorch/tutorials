#!/bin/bash

set -ex

# Install pip packages
pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com -r ./requirements.txt
