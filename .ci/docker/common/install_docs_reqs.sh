#!/bin/bash
# Based off of https://github.com/pytorch/pytorch/tree/b52e0bf131a4e55cd987176f9c5a8d2ad6783b4f/.ci/docker
set -ex

apt-get update
apt-get install -y gpg-agent

curl --retry 3 -sL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

curl --retry 3 -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list

apt-get update
apt-get install -y --no-install-recommends yarn
yarn global add katex --prefix /usr/local

sudo apt-get -y install doxygen

apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
