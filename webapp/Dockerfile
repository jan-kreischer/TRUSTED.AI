# Set the host's base image
FROM python:3.9

# Specify the username
USER root

# set the working directory in the container
WORKDIR /webapp

# copy the dependencies file to the working directory
#COPY ./webapp/requirements.txt .

# copy the content of the local webapp directory to the working directory
COPY . .

# install dependencies
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "python", "./index.py", "1"]

EXPOSE 8080
