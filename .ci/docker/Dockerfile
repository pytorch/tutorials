ARG BASE_IMAGE
ARG CUDA_VERSION
ARG UBUNTU_VERSIOn
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages
ENV ANACONDA_PYTHON_VERSION=3.10
ENV CONDA_CMAKE yes
ENV DOCS yes
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY ./requirements.txt /opt/conda/
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements.txt

# Install gcc
# ENV GCC_VERSION 9
# COPY ./install_gcc.sh install_gcc.sh
# RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install CUDNN
ENV CUDNN_VERSION=8
COPY ./common/install_cudnn.sh install_cudnn.sh
RUN bash ./install_cudnn.sh && rm install_cudnn.sh

USER jenkins
CMD ["bash"]
