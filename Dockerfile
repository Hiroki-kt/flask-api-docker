FROM python:3.8
USER root

RUN apt-get update
RUN apt-get -y install locales && \
  localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

RUN apt-get install -y libgl1-mesa-dev

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

ARG project_dir=/app
WORKDIR $project_dir

COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
RUN export FLASK_DEBUG=1

CMD ["python", "main.py"]
