# These 2 parameters can be overriden by --build-arg <arg> when running docker build.
ARG DOCKER_IMAGE=308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-cuda9-cudnn7-py3:291
FROM ${DOCKER_IMAGE}
ENV PATH=/opt/conda/bin:${PATH}
RUN sudo apt-get update
RUN sudo apt-get install -y --no-install-recommends unzip p7zip-full sox libsox-dev libsox-fmt-all rsync

COPY /home/circleci/project/. /var/lib/jenkins/workspace

WORKDIR /var/lib/jenkins
RUN rm -rf src \
    && pip install -r requirements.txt

# For Tensorboard. Until 1.14 moves to the release channel.
RUN pip install tb-nightly

# Install two language tokenizers for Translation with TorchText tutorial
RUN python -m spacy download en
RUN python -m spacy download de

# PyTorch Theme
RUN rm -rf src \
    && pip install -e git+git://github.com/pytorch/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme

# pillow >= 4.2 will throw error when trying to write mode RGBA as JPEG,
# this is a workaround to the issue.
RUN pip install sphinx-gallery==0.3.1 tqdm matplotlib ipython pillow==4.1.1

RUN aws configure set default.s3.multipart_threshold 5120MB

# Remove runnable code from tutorials that are not supposed to be run
RUN python remove_runnable_code.py beginner_source/aws_distributed_training_tutorial.py beginner_source/aws_distributed_training_tutorial.py || true

RUN make download
