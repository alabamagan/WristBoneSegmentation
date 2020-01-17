#!/bin/bash
# Fixed variables
CHECKPOINT_INCEPTION=checkpoint_Inception.pt
CHECKPOINT_UNET_2=checkpoint_UNET_Cat_2.pt
CHECKPOINT_UNET_3=checkpoint_UNET_Cat_3.pt
CAT_CSV=category_labels.csv

show_help(){
    echo -e "Copyright (C) 2018-2019 by Lun M. Wong and others. The Chinese University of Hong Kong."
    echo -e ""
    echo -e "README:"
    echo -e " This is a bash script for applying the neural network written according to the paper [1]."
    echo ""
    echo -e " This software is released under the GPL license and the author will not be liable for"
    echo -e " any legal consequences of using this software."
    echo ""
    echo -e "Directory:"
    echo -e " This software work with fixed specific directories of input as follow:"
    echo -e " 1) Inception network      <checkpoint-dir>/${CHECKPOINT_INCEPTION}"
    echo -e " 2) UNET for category 2    <checkpoint-dir>/${CHECKPOINT_UNET_2}"
    echo -e " 3) UNET for category 3    <checkpoint-dir>/${CHECKPOINT_UNET_3}"
    echo ""
    echo -e "Generate files"
    echo -e " Category file         <output-dir>/${CAT_CSV}"
    echo -e " Segmentation files    <output-dir>/*nii.gz"


    echo ""
    echo -e "Usage:"
    echo -e " $0 <optstring> <parameters>"
    echo -e " $0 -o <output-dir> -c <checkpoint-dir> -i <input-dir>"
    echo -e " $0 --output-dir=dir checkpoint=dir --input-dir=dir --checkpoint-dir=dir"
    echo ""
    echo "Options:"
    echo -e " -o, --output-dir      Directory to store output segmentations"
    echo -e " -i, --input-dir       Directory to input images"
    echo -e " -c, --checkpoint-dir  Directory that contains all trained network checkpoints"
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

OPTIONS=hc:o:i:
LONGOPTS=help,checkpoint-dir:,output-dir:,input-dir:

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
CHECKPOINT_DIR=checkpoint-dir
OUTPUT_DIR=output-dir
INPUT_DIR=input-dir

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--checkpoint-dir)
            CHECKPOINT_DIR=$2
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR=$2
            shift
            ;;
        -i|--input-dir)
            INPUT_DIR=$2
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

for i in {$CHECKPOINT_DIR,$OUTPUT_DIR,$INPUT_DIR}
do
    if [[ ! -d $i ]]
    then
        echo "Cannot locate folder $i."
    fi
done

echo "Running classification"
source ./run_env.sh
python main_classification.py --useCUDA --load ${CHECKPOINT_DIR}/checkpoint_Inception.pt -b 80 -o ${OUTPUT_DIR}/${CAT_CSV} ${INPUT_DIR}

if [[ ! -f ${OUTPUT_DIR}/${CAT_CSV} ]]
then
    echo "Something wrong with the classification step, please check your input data."
    return 1
fi

echo "Inference for segmentation"
python main.py --useCUDA --load ${CHECKPOINT_DIR} -b 6 -o ${OUTPUT_DIR} ${INPUT_DIR} -C ${OUTPUT_DIR}/${CAT_CSV}

