# AIHACK

Modules

1. Instance segmentation
2. Text generation
  a. Calculation of distance between segmented objects to the reference point
  b. Calculation of angle between segmented objects to the reference point
  c. Calculation of direction between segmented objects to the reference point
  d. Text generation using the above metrics
3. Text to audio conversion

************************************************
Instructions
************************************************
1. Instance segementation

Step 1: create a conda virtual environment with python 3.6
Step 2: install the dependencies
Step 3: Clone the Mask_RCNN repo
Step 4: install pycocotools
Step 5: download the pre-trained weights
Step 6: Test it

Step 1 - Create a conda virtual environment
we used Anaconda with python 3.6.

run this command in a CMD window
conda create -n MaskRCNN python=3.6 pip

Step 2 - Install the Dependencies
place the requirements.txt in your cwdir
https://github.com/markjay4k/Mask-RCNN-series/blob/master/requirements.txt
run these commands

actvitate MaskRCNN
pip install -r requirements.txt
NOTE: we're installing these (tf-gpu requires some pre-reqs)

numpy, scipy, cython, h5py, Pillow, scikit-image, 
tensorflow-gpu==1.5, keras, jupyter

Step 3 - Clone the Mask RCNN Repo
git clone https://github.com/matterport/Mask_RCNN.git
Step 4 - Install pycocotools
NOTE: pycocotools requires Visual C++ 2015 Build Tools
download here if needed https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
clone this repo
git clone https://github.com/philferriere/cocoapi.git
use pip to install pycocotools
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

Step 5 - Download the Pre-trained Weights
Go here https://github.com/matterport/Mask_RCNN/releases
download the mask_rcnn_coco.h5 file
place the file in the Mask_RCNN directory

Step 6 -
execute : AIHACK/MaskRCNN/samples/alert.ipynb and run it 
********************************************************
2. Text Generation
Libraries : PIL(PILLOW), numpy, pandas, math, csv
execute : AIHACK/Mask_RCNN/samples/text_generation.py
*********************************************************
3. Text to audio
Libraries : gTTS, pandas
execute : AIHACK/Mask_RCNN/samples/txt_to_audo.py
