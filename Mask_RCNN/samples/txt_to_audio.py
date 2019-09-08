#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gtts import gTTS 
import os 
import IPython
import pandas as pd


# ## reading the csv file which contains the text data

# In[14]:


txt_data = pd.read_csv('./final_document.txt',header =None)
txt_data.head()


# ## creating separate audio files for each statement 

# In[16]:


for i in range(0,len(txt_data)):
    mytext = txt_data[0][i]
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    myobj.save("test"+str(i)+".mp3") 
  


# ### creating a single audio file for entire text document (for one frame)

# In[17]:


final_list = open("./final_document.txt", "r").read()
TTS = gTTS(text=str(final_list), lang='en')
TTS.save("final_voice.mp3")

