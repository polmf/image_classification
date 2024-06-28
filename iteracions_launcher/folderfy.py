#####################################################
##This script transforms the directory structure of##
##the MAMe dataset. Within a folder with all images##
##creates train/class1, ..., test/class N.         ##
#####################################################

import os
import shutil

# Define the paths to the directories
# We all samples can be found
source_dir = "/gpfs/scratch/nct_299/MAMe/MAMe_data_256/data_256"
# Where training samples are to be moved to
train_dir = "/gpfs/scratch/nct_299/MAMe/MAMe_data_256/data_256/train"
# Where val samples are to be moved to
val_dir = "/gpfs/scratch/nct_299/MAMe/MAMe_data_256/data_256/val"
# Where test samples are to be moved to
test_dir = "/gpfs/scratch/nct_299/MAMe/MAMe_data_256/data_256/test"

# Read the metadata file and move the each JPEG to the appropriate directory
# This file is available in the official MAMe page
metadata_file = "/gpfs/scratch/nct_299/MAMe/MAMe_metadata/MAMe_dataset.csv"

with open(metadata_file, "r") as f:
	f.readline()  # skip the first line
	for line in f:
		img_file = line.strip().split(",")[0]
		img_class = line.strip().split(",")[1]
		if 'train' in line:
			partition = 'train'
		if 'val' in line:
			partition = 'val'
		if 'test' in line:
			partition = 'test'
		src_path = os.path.join(source_dir, img_file)
		dst_dir = os.path.join(source_dir,partition,img_class, img_file)
		if not os.path.exists(dst_dir):
			os.makedirs(dst_dir)
		dst_path = os.path.join(dst_dir,img_file)
		shutil.move(src_path, dst_path)
