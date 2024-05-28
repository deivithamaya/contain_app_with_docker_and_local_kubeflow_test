FROM python:3.8.13 as base


ARG UID
ARG GID
ARG PASSR
ARG PASSA
USER root
#RUN echo ${PASSR} 
RUN echo 'root:'${PASSR} | chpasswd
#RUN apt-get -y update && apt install -y neovim 
## Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
RUN echo 'app:'${PASSA} | chpasswd

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

COPY ./ /src
RUN chmod -R 777 /src && chmod -R 777 /src/static/uploads

WORKDIR /src

RUN pip install --upgrade pip && pip install -r requirements.txt

USER root

FROM base as test
USER root
RUN ["pytest", "-v", "/src/tests"]

FROM base as dev
USER root
RUN apt-get -y update && apt install -y neovim 
RUN echo "i am developing"

FROM base as build
USER ROOT
#ENTRYPOINT ["echo", "${PASSR}","|", "su"]
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "app:app"]
