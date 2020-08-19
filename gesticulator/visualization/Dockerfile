FROM python:3.7
RUN apt-get update
RUN apt-get install -y ffmpeg
WORKDIR /workspace
COPY . .
RUN pip install -e .
