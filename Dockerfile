FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Leonardo Valeriano Neri <lvnufpe@gmail.com>

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Make port 80 available to the world outside this container
EXPOSE 80
