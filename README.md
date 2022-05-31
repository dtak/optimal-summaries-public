# Learning Optimal Summaries of Clinical Time-series with Concept Bottleneck Models

## Installation and Requirements

* ```conda env create -f environment.yml```
* ```conda activate myenv```

## Data Source

The MIMIC-III data source used in these files was created by running [this notebook]().

## Taxonomy

**Before running any other files** in this directory, you first must run the [vasopressor_preprocess_data.ipynb notebook](https://github.com/db769/timeseries-optimal-summaries/blob/main/vasopressor/vasopressor_preprocess_data.ipynb).  This notebook stores standardized data files to a data/ subdirectory.

### Models and helpers
* ```models.py``` contains all of the model parameters and training code for the custom model classes (logreg and LSTM).
* ```custom_losses.py``` contains all of the custom loss functions used during model training.
* ```preprocess_helpers.py``` contains helper functions to load and pre-process patient data.
* ```param_initializations.py``` contains helper functions used to initialize model parameters.
* ```weights_parser.py``` contains the WeightsParser class, which is used to easily match neural network weights to their corresponding feature names and summary functions.

### Scripts
* ```gpu_*```
    * ```gpu_greedy_top_concepts.py``` -- script for greedy optimization method for feature selection within concepts
    * ```gpu_greedy_top_concepts_baseline.py``` -- script for greedy optimization for feature selection in Johnson et al. model
    * ```gpu_lstm_baseline_compound.py``` -- script for training LSTM model using hyperparameters parsed from command-line arguments
    * ```gpu_vasopressor_baseline.py``` -- script for training Johnson et al. logistic regression model using hyperparameters parsed from command-line arguments
    * ```gpu_vasopressor_bottleneck.py``` -- script for training concept bottleneck models using hyperparameters parsed from command-line arguments
    * ```gpu_vasopressor_bottleneck_completeness.py``` -- script for training concept bottleneck models with concept scores in order to calculate completeness score (per Yeh et al.)
    * ```gpu_vasopressor_bottleneck_finetune.py``` -- script for finetuning a trained concept bottleneck models but only using the top k features
    * ```gpu_weights_top_concepts_baseline.py``` -- script for feature selection optimization based on coefficient magnitude for Johnson et al. model
    
* ```launch_*```
    * ```launch_greedy_baseline_jobs.py``` -- launches GPU jobs for ```gpu_greedy_top_concepts.py```
    * ```launch_greedy_bottleneck_jobs.py``` -- launches GPU jobs for ```gpu_greedy_top_concepts_baseline.py```
    * ```launch_lstm_jobs.py``` -- launches GPU jobs for ```gpu_lstm_baseline_compound.py```
    * ```launch_vasopressor_baseline_jobs.py``` -- launches GPU jobs for ```gpu_vasopressor_baseline.py```
    * ```launch_vasopressor_bottleneck_jobs.py``` -- launches GPU jobs for ```gpu_vasopressor_bottleneck.py```
    * ```launch_weights_baseline_jobs.py``` -- launches GPU jobs for ```gpu_weights_top_concepts_baseline.py```
    
### Jupyter Notebooks

* ```model-verification.ipynb``` contains the code which aggregates data from experiments and plots figures/graphs seen in paper
