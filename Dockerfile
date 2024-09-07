FROM ubuntu:20.04

WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PROJ_DIR=/usr


RUN apt-get update && apt-get upgrade -y && \
apt install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt install -y python3.7 && \
apt-get install -y python3-pip libglib2.0-0 && \
apt-get install -y python3-wheel && \
apt-get install -y python3-setuptools && \
apt-get install -y python3-pip && \
apt-get install -y python3-opengl && \
apt-get install -y python3.7-distutils python3-apt && \
apt-get install -y libsm6 libxext6 libxrender-dev && \
python3.7 -m pip install --upgrade pip  && \
python3.7 -m pip install --upgrade einops  && \
python3.7 -m pip install Cython==0.29.37  && \
python3.7 -m pip install numpy==1.19.3 --no-build-isolation

COPY . /app/

RUN groupadd -r www && useradd -r -g www www && \
mkdir /home/www  && \
chown -R www:www /home/www  && \
chown -R www:www /app

USER www
RUN python3.7 -m pip install -r requirements.txt

EXPOSE 8540

CMD ["python3.7", "/app/server.py"]