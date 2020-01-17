# Fully Automated Segmentation of Wrist Bone on T2-weighted fat-suppressed MR Images in Early Rheumatoid Arthritis

## Introduction

Detetecting rheumatoid arthritis (RA) at an early stage is crucial to the progression management of the disease and the prevention of bone erosion. The RA-MRI scoring system (RAMRIS), which measures the inflammation in a semi-quantitative fashion, is currently the standard method to quantify the degree of RA in early stages. To achieve fully quantitative scoring for actual clinical application, it is necessary to realize an automatic solution to extract the region of wrist bones from early MR images of RA patients. 

However, the fully automatic segmentation of wrist bone in RA patients had proven to be a challenging task in T2-weighted fat-suppressed image as the presence of inflammation (manifest as bright clusters on T2W-FS) within the wrist bone (dark on T2W-FS) presents conflicted imaging features that obstruct the development of a reliable segmentation algorithm.   

This repository host the replication of the code implemented and employed in the article [1] that attempts to propose an automatic solution regarding this specific problem. 

## Prerequisite

- torch>=1.2.0
- numpy>=1.17
- SimpleITK>=1.2.0
- scikit-image>=0.15.0
- pandas>=0.20
- configparser>=3.7.0




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
 
#### Format of the csv file

|ID|Type A|Type B|Type C|
|---|---|---|---|
|1|1-2_18-20|3-4_11_16-17|5-10_12_15|
|2| ...|...|...|
 

## References
1. Wong LM, Shi L, Xiao F, Griffith JF. Fully automated segmentation of wrist bones on T2-weighted fat-suppressed MR images in early rheumatoid arthritis. Quant Imaging Med Surg. 2019;9(4):579–589. doi:10.21037/qims.2019.04.03"

