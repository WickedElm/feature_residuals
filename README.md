# Improving Network Intrusion Detection Using Autoencoder Feature Residuals
This repo contains the code needed to support the paper titled "Improving Network Intrusion Detection Using Autoencoder Feature Residuals".
Note that this code contains dependencies on external services such as Weights and Biases and other python libraries.

# Executing Code
Assuming all of the dependencies are in place such as an account on Weights and Biases one can execute experiments by each dataset or all at once.

## Running an experiment for all datasets
To do this simply clone the repo and change to its directory and execute:

```
./run_all_tests
```

This will download the needed data and execute what we considered a single experiment in the paper.

## Running an experiment for a specific datasets
To do this, clone the repo and change to its directory and execute:

```
./download_data
./run_<dataset to run>
```

where you replace <dataset to run> with one of the dataset run scripts in the repo such as run_ctu13_scenario_5
