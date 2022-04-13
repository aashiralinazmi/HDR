#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytesseract as tess
import os
from gtts import gTTS


# In[2]:


tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
value = Image.open("sample_image.jpg")
text = tess.image_to_string(value)
print(text)


# In[3]:


language='en'
output = gTTS(text=text, lang=language, slow=False)
output.save("output.mp3")
os.system("start output.mp3")


# In[ ]:




