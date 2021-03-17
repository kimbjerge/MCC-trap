# Moths Classification and Counting (MCC) 
## What is this repository for? ##

This repository contains all the necessary code and documentation of the algorithm for the light trap and computer vision system to detect, track and classify live moths.
It contains instructions and code for both training and data-processing.

Paper:
https://www.mdpi.com/1424-8220/21/2/343

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

#### Results & output ####
The algorithm outputs the results in JSON and CSV files with date and counts for each species (class).
These statistic files are by default named statistics.json and statistics.csv. 

The statistics.csv file contains information on date and the counted number of moths and insects in the below listed order of categories:

date, 
agrotis_puta, 
amphipyra_pyramidea, 
autographa_gamma, 
caradrinina, 
mythimna_pallens, 
noctua_fimbriata, 
noctua_pronuba, 
xestia_c-nigrum, 
vespula_vulgais, 
background, 
unknown, 
total

Here backgound is the number of blobs the algorithm has identified as part of the background image and unknown is number of unknown insects.
Total contains the total sum of detections in all categories.

Track files are by default named <DirectoryName>.json and <DirectoryName>.csv
The track files contain the following information:

| Property | Description | Example |
|--------------|----------|----------|
| id | The id of the track. | 0 |
| startdate | The date when the track was first registered. YYYY:MM:DD | 20190901 |
| starttime  | The time of the day the track was first registered. HH:MM:SS | 03:32:12 |
| endtime | The time the track was last registered. HH:MM:SS | 03:33:08 |
| duration | The duration of the track in seconds. | 56.00 |
| class | The class predicted by the algorithm. | noctua_pronuba |
| counts | The number of times the given track has been present in a frame | 28.0 |
| confidence | The algorithms confidence in the classification. The confidence is based on the mutual classifications of the track and is calculated as the ratio between the most classified class and the total number of classifications. | 6/10 = 60.00 |
| size | The average number of blob pixels in one track. | 73563.79 |
| distance | The euclidean distance in pixels the centerpoint of the blobs have moved throughout a track. | 65 | 



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
