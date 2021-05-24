FROM mcr.microsoft.com/vscode/devcontainers/python:3.8

COPY requirements.txt /tmp/pip-tmp/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install git gcc unzip make -y \
    && pip3 install --disable-pip-version-check --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
