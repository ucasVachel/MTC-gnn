# MTC-gnn

## Acknowledgments
*Tianjin Intelligent Manufacturing Special Fund Project, No.20201198.*


configs ：
The configuration files in configs are used to set experimental data sets, missing data and hyperparameters

Datasets：
The dataset folder contains files from Datasets at https://doi.org/10.48550/arXiv.1707.01926.

data:
The data folder contains the data processing code, which can complete the generation of data with different missing ratios and the division of training sets, validation sets, and test sets.

Run main_slmgnn.py to start to train.

Before training, make sure to complete the target missing data generation and data partitioning tasks.

//It is worth noting that this model was originally called slmgnn because some changes suggested by the editor ended up being called mtc-gnn

## update
update the weight and code of the relevant baseline
