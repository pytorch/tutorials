ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Setup user
# TODO: figure out how to remove this part
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

COPY ./common/install_pip_requirements.sh install_pip_requirements.sh
COPY ./requirements.txt requirements.txt
RUN bash ./install_pip_requirements.sh && rm install_pip_requirements.sh

RUN ln -s /usr/bin/python3 /usr/bin/python

USER ci-user
CMD ["bash"]
