FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get -y install git vim sudo curl wget apt-transport-https ca-certificates gnupg \
                       libgl1 libglib2.0-0 tmux \
                       python3.9 python3-distutils python3-pip python3-apt cmake swig && \
    rm -rf /var/lib/apt/lists/*

ARG USERID
ARG USERNAME
ARG SERVERNAME
RUN useradd -u $USERID --create-home $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    adduser $USERNAME sudo

USER $USERNAME

ADD requirements.txt .
RUN PATH="${HOME}/.local/bin:${PATH}" python3 -m pip install -r requirements.txt

ADD .tmux.conf .
RUN cp .tmux.conf $HOME/.tmux.conf

WORKDIR /data/$SERVERNAME/$USERNAME
