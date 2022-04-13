#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required modules
import numpy as np
import numpy as imutils
import imutils
import cv2 as cv
import joblib


# In[2]:


#reading image
img = cv.imread('sample_image.jpg')
#resizing image
img = imutils.resize(img,width=300)
#showing original image
cv.imshow("Original",img)
cv.waitKey(0); cv.destroyAllWindows(); cv.waitKey(1)


# In[3]:


#converting image to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#showing grayscale image
cv.imshow("Gray Image",gray)
cv.waitKey(0); cv.destroyAllWindows(); cv.waitKey(1)


# In[4]:


#creating a kernel
kernel = np.ones((40,40),np.uint8)


# In[5]:


#applying blackhat thresholding
blackhat = cv.morphologyEx(gray,cv.MORPH_BLACKHAT,kernel)


# In[6]:


#applying OTSU's thresholding
ret,thresh = cv.threshold(blackhat,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


# In[7]:


#performing erosion and dilation
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)


# In[8]:


#finding countours in image
cnts,hie = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)


# In[9]:


#loading our ANN model
model = joblib.load('model.pkl')
for c in cnts:
    try:
        #creating a mask
        mask = np.zeros(gray.shape,dtype="uint8")
        
    
        (x,y,w,h) = cv.boundingRect(c)
        
        hull = cv.convexHull(c)
        cv.drawContours(mask,[hull],-1,255,-1)    
        mask = cv.bitwise_and(thresh,thresh,mask=mask)

        
        #Getting Region of interest
        roi = mask[y-7:y+h+7,x-7:x+w+7]       
        roi = cv.resize(roi,(28,28))
        roi = np.array(roi)
        #reshaping roi to feed image to our model
        roi = roi.reshape(1,784)

        #predicting
        prediction = model.predict(roi)
    
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv.putText(img,str(int(prediction)),(x,y),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)
        
    except Exception as e:
        print(e)
        
img = imutils.resize(img,width=500)


# In[ ]:


#showing the output
cv.imshow('Detection',img)
cv.imwrite('result.jpg',img)
cv.waitKey(0); cv.destroyAllWindows(); cv.waitKey(1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




