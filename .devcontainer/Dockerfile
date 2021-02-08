FROM python:3.6-slim

COPY requirements.txt requirements.txt

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install git gcc unzip make -y \
    && pip install --no-cache-dir -r requirements.txt