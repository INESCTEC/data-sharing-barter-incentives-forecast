
# Market Sessions Simulator

__NOTE__: This example runs decoupled from Data Market REST-API and database

## Overview

The following directory structure should be considered:

``` bash
.   # Current directory
├── README.md
├── session_sim_normal.py  # main script to execute simulations
├── SimulationManager.py  # configs / reports manager
├── files
|──── datasets # directory for custom datasets
|──── reports  # directory for market runs reports

```

## Before you start

It is important to first install the project requirements. 
To do so, run the following command, in the project ROOT directory:

``` bash
pip install -r requirements.txt
```

__WARNING__: You must use Python 3.8. Also, if you are 
using a virtual environment, make sure it is activated before running the command above.


## Running the simulation

To run the simulation, simply run the following command:

``` bash
python session_sim_normal.py
```

By default simulation will run for 10 sessions, on a hourly basis (`session_freq = 1`), with an example dataset composed by 3 agents, 
each with one resource (see `files/datasets/example_1` directory). 

__NOTE__: By resource we refer to wind farms, solar farms, 
loads, etc. that an agent might own in their portfolio.

On the end of the simulation runs, some report files will be produced and stored in `files/reports/<dataset_name>` directory.
These are:

1. `buyers.csv`: Includes market session buyers information, per resource in their portfolio. 
    It includes:
    * Estimated gain (function and value) by using market forecasts (see `gain_func` and `gain` columns)
    * Initial and final bids (see `initial_bid` and `final_bid` columns).
      * Initial bid is the bid that the agent would have made if it had no information about the market.
      * Final bid is the bid is the initial bid value adjusted according to the potential gain and the `max_payment` value initially defined by the user.
    * Maximum payment that the agent is willing to pay (see `max_payment` column).
    * Final amount that the agent has to pay for this resource

2. `forecasts.csv`: Forecasts produced by the market model, for each buyer resource.
3. `sellers.csv`: Includes market session sellers information, per resource in their portfolio. 
    It includes:
    * Final amount that the agent has to receive for this resource

__NOTE__: The simulation can take a while to run, depending on the number of sessions and resources in the dataset. Also, the final CSV's can be merged according to the market session or user / user resource identifiers.


## Using your own datasets

To use your own datasets, simply add them to `files/datasets` directory. However, the market will only run when you have the following information:

1. `files/<dataset_name>/dataset.csv`: Dataset, disposed in tabular form with 1 column per user resource and a `datetime` column for the timestamp references (in universal timezone).
2. `files/<dataset_name>/user_resources.json`: JSON file mapping each user identifier with its resources (direct map with the columns in `dataset.csv`).
3. `files/<dataset_name>/bids/<scenario>/bids.json`: Bids scenarios for this dataset. These will be the bids the buying agents will make in each market session.

Please analyze the `files/datasets/example_1` directory for an example of how to structure your dataset.

