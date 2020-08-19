FROM python:3.6.7

# set a directory for the app
WORKDIR /workspace

# copy all the files to the container
COPY . .

# install dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg
RUN pip install --upgrade pip
RUN apt-get -y install libsndfile-dev
RUN pip install -r gesticulator/requirements.txt
RUN pip install -e .
RUN pip install -e gesticulator/visualization