# DetectionSystem

A system to detect vehicles that are moving based on a series of empirical steps. 

The final detections are done using 3 approaches:

   - Region proposal + Hand-engineered SVM with more features
   - Region proposal + Hand-engineered SVM with lesser features
   - Region proposal + YOLOv3


## run.sh

A shell script to make execution of detectionsystem.py much simpler and abstracted.
The help (-h) parameter can guide you through all the available options. 

Sample:
```sh
$ sudo chmod +x run.sh
$ ./run.sh -h

Run this script with -r flag to execute all approaches with results validation enabled

Syntax: Template [-c|p|t|f|s|v]

options:

c     0 = Slow SVM, 1 = Fast SVM, 2 = Modified YOLO, 3 = Orginal YOLO
p     0 = For normal processing, 1 = For parallel processing
t     Validate/test results
f     Visualize/Display Frames
s     Visualize/Display Sub-Frames/Regions
v     Verbose mode.
k     Save/keep as file.

Default Option values -c 2 -p 0 -t False -f False -s False -v False
```
## dependencies.sh
Script to check if all the dependencies are downloaded, and download them if not.

Unresolved file dependencies - nuscenes mini/full dataset needs to be downloaded from nuscenes [website](https://www.nuscenes.org/). 
Not sharing it or making it availabe through a `wget`in `dependencies.sh` due to Terms of use.

## detectionsystem.py

Main file which controls the detections / predictions of all the moving vehicles. Takes in the same arguments as `run.sh`, and has a similar default option values.  

## densityscan.py

Clustering module which can cluster features based on different feature distances

## SVCmodel.ipynb

File containing the training of hand-engineered SVM classifier models (Fast and Slow) which are done by varying the feature extraction techniques.

## requirements.txt

Use the below command to install all the dependencies, or run `dependencies.sh` for executing it. 

```sh
$ pip3 install -r requirement.txt 
``` 
 
