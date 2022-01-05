FROM python:3.8
USER root

RUN apt-get update

ARG project_dir=/app/

WORKDIR $project_dir

RUN pip install flask

CMD ["python", "main.py"]
