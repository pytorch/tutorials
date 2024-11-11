#!/bin/bash
# Based off of https://github.com/pytorch/pytorch/tree/b52e0bf131a4e55cd987176f9c5a8d2ad6783b4f/.ci/docker

set -ex

install_ubuntu() {
  # Install common dependencies
  apt-get update
  # TODO: Some of these may not be necessary
  apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake=3.22* \
    curl \
    git \
    wget \
    sudo \
    vim \
    jq \
    vim \
    unzip \
    gdb \
    rsync \
    libssl-dev \
    p7zip-full \
    libglfw3 \
    libglfw3-dev \
    sox \
    libsox-dev \
    libsox-fmt-all \
    python3-pip \
    python3-dev

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
