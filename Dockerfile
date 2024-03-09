FROM python:3.12-bookworm

#change working directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  # Poetry's configuration:
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local'

# Accept GITLAB_TOKEN as a build-time argument
ARG GITLAB_TOKEN

# Use the argument to set the Git configuration for HTTPS cloning
RUN git config --global url."https://oauth2:${GITLAB_TOKEN}@gitlab.inesctec.pt/".insteadOf "https://gitlab.inesctec.pt/"

RUN apt-get update && apt-get install -y build-essential

# install required packages
# copy requirements
COPY poetry.lock pyproject.toml /app/
RUN pip install poetry && poetry install

# copy project
COPY . /app

#CMD ["tail", "-f", "/dev/null"]
