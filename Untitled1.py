#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-python-headless')
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


import os
Datadirectory = r"D:\Study\4-1\thesis\emotion\Dataset\train"
print(os.path.exists(Datadirectory))  # Should return True
 # Should return True if the directory exists
classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]


# In[7]:


for category in classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break  # Exit the inner loop after displaying the first image
    break  # Exit the outer loop after processing the first category


# In[9]:


img_size=224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()


# In[11]:


new_array.shape


# In[13]:


training_Data = []
def create_training_Data():
    for category in classes:
        path = os.path.join(Datadirectory, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


# In[15]:


create_training_Data()


# In[16]:


print(len(training_Data))


# In[23]:


import random
random.shuffle(training_Data)


# In[33]:


x = []  # Initialize as a list
y = []  # Initialize for labels

for features, label in training_Data:
    x.append(features)  # Append features to the list
    y.append(label)     # Append labels to the list

x = np.array(x).reshape(-1, img_size, img_size, 3)  # Convert to NumPy array and reshape


# In[35]:


x.shape


# In[41]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0/255.0)
# Define your data directory and preprocess data in real time
train_generator = datagen.flow_from_directory(
    r"D:\Study\4-1\thesis\emotion\Dataset\train",  # Replace with your data directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# In[47]:


Y= np.array(y)
Y.shape


# In[51]:


# deep learning model for training _ transfer learning

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[53]:


model = tf.keras.applications.MobileNetV2()


# In[55]:


model.summary()


# In[77]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# Load the pre-trained MobileNetV2 model without the top classification layer
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Base model input and output
base_input = base_model.input  # Input layer of the base model
base_output = base_model.output  # Output layer of the base model

# Add custom layers to the model
final_output = layers.Dense(128)(base_output)  # Add a dense layer
final_output = layers.Activation('relu')(final_output)  # Add activation
final_output = layers.Dense(64)(final_output)  # Add another dense layer
final_output = layers.Activation('relu')(final_output)  # Add activation
final_output = layers.Dense(7, activation='softmax')(final_output)  # Final classification layer

# Create the new model
new_model = keras.Model(inputs=base_input, outputs=final_output)

# Print the model summary
new_model.summary()


# In[79]:


new_model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])


# In[ ]:


new_model.fit(x,Y,epochs = 15)


# In[71]:





# In[73]:


final_output


# In[ ]:




