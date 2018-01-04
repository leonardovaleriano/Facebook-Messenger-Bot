FROM tensorflow/tensorflow:latest-py3
MAINTAINER Leonardo Valeriano Neri <leonardovalerianoneri@gmail.com>

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Make port 80 available to the world outside this container
EXPOSE 80
