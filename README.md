# Fully Automated Segmentation of Wrist Bone on T2-weighted fat-suppressed MR Images in Early Rheumatoid Arthritis

## Introduction

Detetecting rheumatoid arthritis (RA) at an early stage is crucial to the progression management of the disease and the prevention of bone erosion. The RA-MRI scoring system (RAMRIS), which measures the inflammation in a semi-quantitative fashion, is currently the standard method to quantify the degree of RA in early stages. To achieve fully quantitative scoring for actual clinical application, it is necessary to realize an automatic solution to extract the region of wrist bones from early MR images of RA patients. 

However, the fully automatic segmentation of wrist bone in RA patients had proven to be a challenging task in T2-weighted fat-suppressed image as the presence of inflammation (manifest as bright clusters on T2W-FS) within the wrist bone (dark on T2W-FS) presents conflicted imaging features that obstruct the development of a reliable segmentation algorithm.   

This repository host the replication of the code implemented and employed in the article [1] that attempts to propose an automatic solution regarding this specific problem. 

## License

Copyright (c) 2019-2020 Lun M. Wong.

The use of the this software should be abided by the MIT License.

## Prerequisite

- torch>=1.2.0
- numpy>=1.17
- SimpleITK>=1.2.0
- scikit-image>=0.15.0
- pandas>=0.20
- configparser>=3.7.0


## Using with Docker

This repo has published a docker image, which can be used with nvidia-docker. Make sure you have nvidia-docker installed so that you can directly use the docker image. You can follow [this link](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster) for the installation steps.

Download the package image:
```bash
docker pull docker.pkg.github.com/alabamagan/wristbonesegmentation/wbs_docker:1.0.2
```

Please use the following command on a machine with at least 12gb GRAM:
```bash
# -v suggest your host /your/data directory is now connected to /Data of the container
docker run -v /your/data:/Data --ipc=host --gpus all --ipc=host -it wrist_bone_segmentation
```  

You are now connected to the container session and you can go to the directory ```/root/Source/WristBoneSegmentation``` for your own application.

## Usage

The algorithm is separated into two steps, which has to be trained separately, the classification step and the segmentation step. 

### Training


1. Put all training materials in to one folder, the images should all be stored in NIFTI format in folder ```./Data/Training```
2. Prepare a ground-truth category csv file that is written in the format written below in the file ```./Data/Training/category_labels.csv```
3. Start training by command:

#### Classification Step 
 ```bash
 # cd to this directory
 ./train.sh -o [dir] -c [saved cp (optional)] -s 0 
 ```
  
#### Segmentation Step

Note that the name of checkpoints in this step is fixed to ```checkpoint_UNET_Cat_2.pt``` and ```checkpoint_UNET_Cat_3.pt```. Both checkpoints needs to be present inorder to load.
```bash
 # cd to this directory
 ./train.sh -o [dir] -c [saved cp DIRECTORY (optional)] -s 1 
 ```
 
 ### Inference
 
 After you have trained the networks, you can proceed to perform inference. Note that if you downloaded the docker image, you will found the networks to be already trained and were situated in the directory ```<source_root>/Backup```.
 
 This time, there is no need to separate the process into two, simply use the following command:
 
 ```bash
./inference.sh -o [dir] -c [dir] -i [dir]
```
The output will be generated to the directory specified by the -o option, with each the same file name as the input.

### Help

 Using the ```-h``` option in both the script ```inference.sh``` and ```train.sh``` will allow you know more about the functionality of the algorithm.  
 
 
### Format of the csv file

|ID|Type A|Type B|Type C|
|---|---|---|---|
|1|1-2_18-20|3-4_11_16-17|5-10_12_15|
|2| ...|...|...|
 

## References
1. Wong LM, Shi L, Xiao F, Griffith JF. Fully automated segmentation of wrist bones on T2-weighted fat-suppressed MR images in early rheumatoid arthritis. Quant Imaging Med Surg. 2019;9(4):579–589. doi:10.21037/qims.2019.04.03"

