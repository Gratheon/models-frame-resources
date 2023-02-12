FROM ubuntu:23.04

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PROJ_DIR=/usr

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip python3-dev

# Install dependencies
RUN apt-get install -y git sudo wget lsb-release software-properties-common

# Install build-essential
RUN apt-get install -y build-essential curl

# Install systemd
RUN apt-get install -y systemd systemd-sysv

# Install python libs
RUN apt-get install -y python3-wheel python3-setuptools python3-pip
RUN apt-get install -y python3-paho-mqtt python3-logzero python3-astor
RUN apt-get install -y python3-opengl python3-six python3-grpcio
RUN apt-get install -y python3-keras-applications python3-keras-preprocessing
RUN apt-get install -y python3-protobuf python3-termcolor python3-numpy

RUN apt-get install -y pkg-config python3-h5py libhdf5-dev
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8540

CMD ["python3", "app.py"]
