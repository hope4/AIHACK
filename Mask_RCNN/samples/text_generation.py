#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
from PIL import Image, ImageDraw
import csv


# ### Reading the csv file in which detections from segmentation module has been stored

# In[3]:


data = pd.read_csv('./output.csv')
data.head()


# ### Reading the bounding box co-ordinates (top-left, bottom-right) and calculating the 5 co-ordinates (4 corners +center)

# In[9]:


for i in range(0,len(data)):
    p = []
    p1 = [data.iloc[i][2], data.iloc[i][1]]
    print(p1)
    p.append(p1)
    p2 = [data.iloc[i][4], data.iloc[i][1]]
    print(p2)
    p.append(p2)
    p3 = [data.iloc[i][2], data.iloc[i][3]]
    print(p3)
    p.append(p3)

    p4 = [data.iloc[i][4], data.iloc[i][3]]
    print(p4)
    p.append(p4)

    height = np.absolute(data.iloc[i][1]- data.iloc[i][3])
    width = np.absolute(data.iloc[i][2]- data.iloc[i][4])
    p5 = [data.iloc[i][2]+ int(height/2), data.iloc[i][1]+int(width/2)]
    print(p5)
    p.append(p5)
    print(p)
### calculating the distance from the reference point (mid point of the visually impaired vehicle) to every object
##(5 co-ordinates) present in the frame. 
    dist = []
    dist1 = math.sqrt(((480-p1[0])**2)+((720-p1[1])**2))
    print(dist1)
    dist.append(dist1)
    dist2 = math.sqrt(((480-p2[0])**2)+((720-p2[1])**2))
    print(dist2)
    dist.append(dist2)
    dist3 = math.sqrt(((480-p3[0])**2)+((720-p3[1])**2))
    print(dist3)
    dist.append(dist3)
    dist4 = math.sqrt(((480-p4[0])**2)+((720-p4[1])**2))
    print(dist4)
    dist.append(dist4)
    dist5 = math.sqrt(((480-p5[0])**2)+((720-p5[1])**2))
    print(dist5)
    dist.append(dist5)
    print (dist)
### the minimum distance among the five co-ordinates is considered
    print (np.argmin(dist))
    a = p[np.argmin(dist)]
    df = [a[0],a[1],min(dist)]
## writing the minimum distance w.r.t every object and the respective bbnd box co-ordinates to csv file
    with open('document.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(df)
    c = [480,720]
    im = Image.open("./directions.png")
    d = ImageDraw.Draw(im)
    line_color = (0, 0, 255)
    point_color = (0,255,0)
    #d.ellipse((103,516,108,522),fill=line_color)
    d.line((a[0],a[1],c[0],c[1]), fill=point_color, width=3)
###drawing the lines to the minimum distance coordinate points
    im.save("./directions.png")
im.show("./directions.png")


# ### Reading the csv file (saved from the previous module) to calculate the angles at which the objects are present

# In[27]:


data1= pd.read_csv('./document.csv',header = None)


# In[64]:


ang = []
for j in range(0,len(data1)):
    adj = np.absolute(int(data1.iloc[j][0])-480)
    print(adj)
    angle = math.degrees(math.acos((adj/data1.iloc[j][2])))
    print (angle)
    ang.append(angle)
df1 = pd.DataFrame(ang)
df1.to_csv('document1.csv',header = None, index = False)


# ### creating the line at the center and the objects based on the bbng box coordinates, that are less than the center are considered left and greater than as right and equal to as the center w.r.t the reference (visually impaired vehicle)

# In[50]:


with open('document2.csv','w') as f1:
    writer=csv.writer(f1)
    for k in range(0,len(data1)):
        if data1.iloc[k][0] < 480:
            print("left")
            l = "left"
            row = l
        elif data1.iloc[k][0] > 480:
            print ("right")
            r = "right"
            row = r
        else:
            print("center")
            c = "center"
            row = c
        writer.writerow([row])


# ### writing to the text file the values (object, distance, angle and left/right) calculated from the above modules

# In[6]:


data = pd.read_csv('./output.csv')
data1 = pd.read_csv('./document.csv',header = None)
data2 = pd.read_csv('./document1.csv',header = None)
data3 = pd.read_csv('./document2.csv',header = None)
#print(len(data),len(data1),len(data2),len(data3))
with open('./final_document.txt','w') as f2:
    writer=csv.writer(f2)
    row1 = "Hello!! Welcome to NVIDIA AI Hackathon demo of team 7. You have the following objects around you."
    writer.writerow([row1])
    for k in range(0,len(data)):
        p = str(data.iloc[k][5])
        q= str(int(data1.iloc[k][2]))
        r =str(int(data2.iloc[k][0]))
        row ="There is a "+ p +" at a distance of "+ q +" pixel units and at an angle of "+ r +" degrees to the "+ str(data3.iloc[k][0]) +" of you."
        writer.writerow([row])
    row2 = "Thank you! Have a safe day."
    writer.writerow([row2])


# In[ ]:




