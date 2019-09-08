#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[ ]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[ ]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# In[ ]:


# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]

import os
import csv

# with open('output.csv', 'w') as writeFile:
#     writer = csv.writer(writeFile)
#     writer.writerows(['Image name','ROI','Masks','Class_ids','Confidence Scores'])
#     for i in files:
#     #     image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#         # Run detection
#         print("Reading file..."+str(i))
#         image = skimage.io.imread('/home/navin/Desktop/AIHACK/Mask_RCNN/imgs/'+str(i))
#         results = model.detect([image], verbose=1)

#         # Visualize results
#         r = results[0]
#     #     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
# #                                     class_names, r['scores'])
#         writer = csv.writer(writeFile)
#         writer.writerows([str(i),r['rois'],r['masks'],r['class_ids'],r['scores']])
with open('output.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(['Image name','ROI_x1','ROI_y1','ROI_x2','ROI_y2','Class_ids','Confidence Scores'])

    for i in files:
        #     image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
            # Run detection
        print("Reading file..."+'0016E5_05940.png')
        image = skimage.io.imread('./'+'0016E5_05940.png')
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        print(str(i),'\n\n',r['rois'],'\n\n',r['masks'],'\n\n',r['class_ids'],'\n\n',r['scores'])
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
#         print(image1)
        for k in range(len(r['rois'])):
            writer = csv.writer(writeFile)
            writer.writerow(['0016E5_05940.png',r['rois'][k][0],r['rois'][k][1],r['rois'][k][2],r['rois'][k][3],class_names[r['class_ids'][k]],r['scores'][k]])
            
        break


# In[82]:


class_names[r['class_ids'][0]]


# In[ ]:


len(r['scores'])


# In[ ]:


import numpy as np
len(np.where(r['masks']==True))


# In[ ]:


len(np.where(r['masks']==True)[2])


# In[ ]:





# In[ ]:




