FROM python:3.8.13 as base

ARG UID
ARG GID
ARG PASSR
ARG PASSA
USER root

RUN echo 'root:'${PASSR} | chpasswd
#RUN apt-get -y update && apt install -y neovim 
# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --gecos '' --uid $UID --gid $GID app
RUN echo 'app:'${PASSA} | chpasswd

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

ADD requirements.txt .
RUN pip3 install -r requirements.txt

#USER app
USER root

ENV PYTHONPATH=$PYTHONPATH:/src/

COPY ./ /src/

WORKDIR /src

FROM base as test
user root
RUN ["pytest", "-v", "/src/tests"]

FROM base as dev
USER ROOT
RUN apt-get -y update && apt install -y neovim 
RUN echo "i am developing"

FROM base as build
USER ROOT
ENTRYPOINT ["python3", "/src/ml_service.py"]

