#!/bin/bash

show_help(){
    echo -e "Copyright (C) 2018-2019 by Lun M. Wong and others. The Chinese University of Hong Kong."
    echo -e ""
    echo -e "README:"
    echo -e " This is a bash script for training the neural network written according to the paper [1]."
    echo -e " There are two individual networks to train, namely the 1) Inception V3 and 2) UNet. The"
    echo -e " first network classify each slices into three categories, the second segment the"
    echo -e " classified slices for two of the three categories."
    echo ""
    echo -e " This software is released under the GPL license and the author will not be liable for"
    echo -e " any legal consequences of using this software."
    echo ""
    echo -e "Directory:"
    echo -e " This software work with specified directories, meaning all the data and network states"
    echo -e " should be stored in specified folders. The specifications is as follow:"
    echo -e " 1) Training data:             ./Data/Training"
    echo -e " 2) Groundtruth categories:    ./Data/Training/category_labels.csv"
    echo -e " 3) Segmentation groundtruth:  ./Data/Segmentation"

    echo ""
    echo -e "Usage:"
    echo -e " $0 <optstring> <parameters>"
    echo -e " $0 -o dir -s {0|1} [-c dir]"
    echo -e " $0 --out-checkpoint=dir --step={0|1} [--checkpoint dir]"
    echo ""
    echo "Options:"
    echo -e " -o, --out-checkpoint   Output directory of checkpoints"
    echo -e " -s, --step {0|1}          0 for classification, 1 for segmentation"
    echo -e " -h, --help                Show this help message"
    echo -e " -c, --checkpoint          [Optional] Load previous checkpoints."
    echo ""
    echo "Reference:"
    echo " [1]  Wong LM, Shi L, Xiao F, Griffith JF. Fully automated segmentation of wrist bones on T2-weighted"
    echo "      fat-suppressed MR images in early rheumatoid arthritis. Quant Imaging Med Surg. 2019;9(4):579–589."
    echo "      doi:10.21037/qims.2019.04.03"
}

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.



# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} != 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=hc:o:s:
LONGOPTS=help,checkpoint:,out-checkpoint:,step:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} != 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

# Initialize our own variables:
CHECKPOINT_DIR=n
OUTPUT_CHECKPOINT_DIR=n
TRAINING_STEP=n

# Fixed variables for relative directories
TRAINING_DATA_DIR=./Data/Training
GROUND_TRUTH_CATEGORIES=./Data/Training/category_labels.csv
SEGMENTATION_GROUND_TRUTH=./Data/Segmentation

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--checkpoint)
            CHECKPOINT_DIR=$2
            shift
            ;;
        -o|--out-checkpoint)
            OUTPUT_CHECKPOINT_DIR=$2
            shift
            ;;
        -s|--step)
            TRAINING_STEP=$2
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            #echo "Ignoring argument inputs $1."
            #exit 3
            shift
            ;;
    esac
done



if [[ $OUTPUT_CHECKPOINT_DIR == n ]];
then
    OUTPUT_CHECKPOINT_DIR="./Backup/"
    echo "Using default output path ${OUTPUT_CHECKPOINT_DIR}"
fi

# Check if directory exist, create otherwise
if [[ ! -d $OUTPUT_CHECKPOINT_DIR && ${TRAINING_STEP} == 1 ]]
then
    mkdir -p $OUTPUT_CHECKPOINT_DIR
fi

# Check if input checkpoint exist
if [[ $TRAINING_STEP == 0 ]]
then
    if [[ ! -f ${CHECKPOINT_DIR} && ! ${CHECKPOINT_DIR} == n ]]
    then
        echo "Cannot find checkpoint/checkpoints under ${CHECKPOINT_DIR}"
        exit 3
    fi
else
    # Check if input checkpoint exist for segmentation
    if [[ ${CHECKPOINT_DIR} == n ]]
    then
        echo "No checkpoint provided, creating new checkpoints..."
    else
        for checkpoint_names in {"checkpoint_UNET_Cat_2.pt","checkpoint_UNET_Cat_3.pt"}
        do
            if [[ ! -f ${CHECKPOINT_DIR}/${checkpoint_names} ]]
            then
                echo "Cannot find ${CHECKPOINT_DIR}/${checkpoint_names}"
                exit 3
            fi
        done
    fi
fi

case $TRAINING_STEP in
    0)
        echo "Training classification network."
        source ./run_env.sh
        python main_classification.py --useCUDA --train ${GROUND_TRUTH_CATEGORIES} -d 0.005 -e 1000 --train-params "{'lr':1E-5,'momentum': 0.9}" -b 80 ${TRAINING_DATA_DIR} --load ${CHECKPOINT_DIR} --checkpoint $OUTPUT_CHECKPOINT_DIR
        ;;
    1)
        echo "Training segmentation networks."
        for i in 1 2
        do
            echo "Training for category $i"
            python main.py --useCUDA --train ${SEGMENTATION_GROUND_TRUTH} -d 0.1 -e 300 --train-params "{'lr':1E-4,'momentum': 0.2}" -b 6 ${TRAINING_DATA_DIR} --load ${CHECKPOINT_DIR} --checkpoint $OUTPUT_CHECKPOINT_DIR/checkpoint_UNET_Cat_$((i + 1)).pt -C ${GROUND_TRUTH_CATEGORIES} -c $i
        done
        shift
        ;;
    *)
        echo "Must specify which step you are training -s {0|1}."
        exit 3
        ;;
esac