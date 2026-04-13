FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake wget curl ca-certificates \
    libopencv-dev python3 python3-pip python3-dev \
    sudo iputils-ping \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -s /bin/bash -m -d /home/vscode vscode && \
    echo vscode:vscode | chpasswd && \
    usermod -aG sudo vscode && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER vscode
RUN pip3 install --upgrade pip setuptools
RUN pip3 install fastapi uvicorn python-multipart numpy opencv-python requests pyyaml easydict onnx

WORKDIR /workspaces/darknet
CMD ["/bin/bash"]