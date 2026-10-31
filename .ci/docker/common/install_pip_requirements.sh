#!/bin/bash

set -ex

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668). Allow
# pip to install into it for every user, including the later `sudo pip` calls.
printf '[global]\nbreak-system-packages = true\n' > /etc/pip.conf

# Install pip packages
# --ignore-installed: the apt-installed pip has no RECORD file to uninstall from.
pip install --upgrade --ignore-installed pip
pip install -r ./requirements.txt
