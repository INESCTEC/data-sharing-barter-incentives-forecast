# VALOREM - Data Market Engine

-----------------------------------------------------

[![version](https://img.shields.io/badge/version-0.0.1-blue.svg)]()
[![status](https://img.shields.io/badge/status-development-yellow.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)

Preliminary documentation available at the project `docs/` directory.


## Project Structure:

The following directory structure should be considered:

``` bash
.   # Current directory
├── conf  # project settings
├── docs  # useful docs
├── examples  # example scripts (includes simulation decoupled from DB + REST)
├── packages  # project packages (i.e., precompiled versions of external packages)
├── src  # project source code
├── .coveragerc  # code coverage configs
├── .flake8  # flake8 configs
├── .gitignore  # gitignore file
├── .gitlab-ci.yml  # gitlab-ci file
├── docker-compose.yml  # docker-compose file
├── Dockerfile  # project dockerfile
├── dotenv  # template for environment variables
├── pytest.ini  # pytest configs
├── README.md
├── requirements.txt  # project dependencies
├── run_menu.py  # interactive menu for running the market
├── run_market_pipeline.py  # script for running the market pipeline
```

## Initial setup:

> **_NOTE:_**  The commands below assume that you are running them from the root directory of the project (`energy_app/`)


### Configure environment variables:

The `dotenv` file provides a template for all the environment variables needed by this project. 
To configure the environment variables, copy the `dotenv` file to `.env` and fill in the values for each variable.

```shell
   $ cp dotenv .env
```
**_NOTE:_** In windows, just copy-paste the `dotenv` file and rename it to `.env`.


### With Docker:

To launch the docker containers stack:

```shell
   $ docker compose build
```

**_NOTE:_**  This will create the market image, which will be then executed later


### With Local Python Interpreter:

If you prefer using your local python interpreter (instead of docker), you'll need to manually perform the installation steps.
Also, only 'simulation' functionalities (i.e., without integration with the data market REST / DB) will be available.

1. Install the python dependencies
   ```shell
        $ pip install -r requirements.txt
   ```

2. Run the 'run_menu.py' script to open the interactive market menu
    ```shell
        $ python run_menu.py
    ```

### How to run:

> **_NOTE 1:_**  The following instructions assume that the data market database and REST API are already initialized (available in other projects).
> **_NOTE 2:_**  The commands below assume that you are running them from the root directory of the project (`energy_app/`)

Run the market interactive menu:

#### With Docker:

```shell
   $ docker compose run --rm app python run_menu.py
```

#### With local interpreter:
    
```shell
    $ python run_menu.py
  ```


## Contacts:

If you have any questions regarding this project, please contact the following people:

Developers (SW source code / methodology questions):
  - José Andrade <jose.r.andrade@inesctec.pt>
  - André Garcia <andre.f.garcia@inesctec.pt>

Contributors / Reviewers (methodology questions):
  - Carla Gonçalves <carla.s.goncalves@inesctec.pt>
  - Ricardo Bessa <ricardo.j.bessa@inesctec.pt>
