FROM python:3.8-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PROJ_DIR=/usr

# System dependencies for OpenCV and scientific stack (h5py/SciPy build requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 build-essential \
    pkg-config libhdf5-dev gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy only requirements first (leverage Docker cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy rest of application
COPY . /app/

RUN groupadd -r www && useradd -r -g www www && \
    mkdir /home/www  && \
    chown -R www:www /home/www  && \
    chown -R www:www /app

USER www

EXPOSE 8540

CMD ["python", "/app/server.py"]
