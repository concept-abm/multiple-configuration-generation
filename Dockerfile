FROM docker.io/library/python:3.11-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY generate-config.py generate-config.py

ENTRYPOINT [ "python3", "/app/generate-config.py"]
