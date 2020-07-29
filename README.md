# Moths Classification and Counting (MCC) 
## What is this repository for? ##

This repository contains all the necessary code and documentation of the algorithm for the light trap and computer vision system to detect, track and classify live moths.
It contains instructions and code for both training and data-processing.

Paper:
https://www.biorxiv.org/content/10.1101/2020.03.18.996447v1


## How do I get started? ##
#### Dependencies ####
The following dependencies must be installed.

| Dependency   | Version  |
|--------------|----------|
| scikit_image | 0.16.2	  |
| numpy        | 1.18.5   |
| scipy        | 1.4.1    |
| Pympler      | 0.7      |
| tensorflow   | 2.0.0    |
| Pillow       | 7.0.0    |
| pandas       | 1.0.4    |
| Seaborn      | 0.10.1   |
| Scikit learn | 0.19.1   |

#### Using Anaconda: ####
1. Install the dependencies and create the environment using the provided OS specific environment file with the command "conda create --name myEnv --file ENV_FILE.txt"
   (See env file for Linux and Windows)
2. Activate the enviorement using the command "activate myEnv"

#### Start the program ####
Start the programs by running the files MCC_gui.py or MCC_algorithm.py in the code directory with the command "python MCC_gui.py" or "python MCC_algorithm.py".

## How do I train new models? ##
New models can be trained using the provided python script: code/hp_param_training.py 

Extract the 10classes_mixed.zip file that contains the training and validation dataset.

This script is configured for 10 classes including one for background images.

To train a model with your own dataset edit the hp_param_training.py script by providing a data path, 
logging path, model save path and edit the steps per epoch based on the size of the dataset.

Tensorboard command: tensorboard --logdir hparam_tuning --reload_multifile=true

## Who do I talk to? ##
Jakob Bonde Nielsen or Kim Bjerge

Email: jakob.bonde.nielsen@gmail.com, kbe@ase.au.dk
