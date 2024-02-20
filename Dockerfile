FROM python:3.12-bookworm

#change working directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Accept GITLAB_TOKEN as a build-time argument
ARG GITLAB_TOKEN

# Use the argument to set the Git configuration for HTTPS cloning
RUN git config --global url."https://oauth2:${GITLAB_TOKEN}@gitlab.inesctec.pt/".insteadOf "https://gitlab.inesctec.pt/"
# copy requirements
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y build-essential

# install required packages
RUN pip install -r requirements.txt

# copy project
COPY . /app

#CMD ["tail", "-f", "/dev/null"]
