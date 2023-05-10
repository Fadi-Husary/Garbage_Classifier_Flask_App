#!/usr/bin/env python
# coding: utf-8
 
# In[1]:


pip install flask tensorflow keras pillow


# In[2]:


from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io


# In[3]:


import numpy as np
import pandas as pd 
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras.applications.xception as xception
import zipfile
import sys
import time
import tensorflow.keras as keras
import tensorflow as tf
import re

from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.models import Model, Sequential

from keras.utils import to_categorical


# In[4]:


IMAGE_WIDTH = 320    
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

path = "Desktop/garbage_classification/"
categories = {1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash', 6: 'battery',
              7: 'shoes', 8: 'clothes', 9: 'green-glass', 10: 'brown-glass', 11: 'white-glass',
              12: 'biological'}


# In[5]:


def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
    return df


# In[6]:


filenames_list = []
categories_list = []


# In[7]:


for category in categories:
    filenames = os.listdir(path + categories[category])
    
    filenames_list = filenames_list  +filenames
    categories_list = categories_list + [category] * len(filenames)
    
df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

print('number of elements = ' , len(df))


# In[8]:


df.head()


# In[9]:


random_row = random.randint(0, len(df)-1)
sample = df.iloc[random_row]
randomimage = image = tf.keras.utils.load_img(path +sample['filename'])
print(sample['filename'])
plt.imshow(randomimage)


# In[10]:


df_visualization = df.copy()
# Change the catgegories from numbers to names
df_visualization['category'] = df_visualization['category'].apply(lambda x:categories[x] )

df_visualization['category'].value_counts().plot.bar(x = 'count', y = 'category' )

plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)
plt.title("Count of images per class", y=1.02);


# In[11]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import keras.applications.xception as xception


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


from keras.layers import Lambda


# In[14]:


xception_layer = xception.Xception(include_top = False, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
                       weights = 'Downloads/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

# We don't want to train the imported weights
xception_layer.trainable = False


sqnl = Sequential()
sqnl.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

#create a custom layer to apply the preprocessing
def xception_preprocessing(img):
    return xception.preprocess_input(img)

sqnl.add(Lambda(xception_preprocessing))

sqnl.add(xception_layer)
sqnl.add(tf.keras.layers.GlobalAveragePooling2D())
sqnl.add(Dense(len(categories), activation='softmax')) 

sqnl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

sqnl.summary()


# In[15]:


early_stop = tf.keras.callbacks.EarlyStopping(patience = 2, verbose = 1, monitor='val_categorical_accuracy' , mode='max', min_delta=0.001, restore_best_weights = True)

callbacks = [early_stop]


# In[16]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[17]:


df["category"] = df["category"].replace(categories) 

# We first split the data into two sets and then split the validate_df to two sets
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print('train size = ', total_validate , 'validate size = ', total_validate, 'test size = ', test_df.shape[0])


# In[18]:


batch_size=64

train_datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip = True,
    width_shift_range=0.2,
    height_shift_range=0.2
    
) 

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    path, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[19]:


validation_datagen = ImageDataGenerator()

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    path, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[20]:


EPOCHS = 20
history = sqnl.fit_generator(
    train_generator, 
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[36]:


sqnl.save('gc.h5')


# In[37]:


sqnl.summary()


# In[22]:


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend()

ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
ax2.legend()

legend = plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[23]:


test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe= test_df,
    directory=path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=False 
)


# In[29]:


filenames = test_generator.filenames
nb_samples = len(filenames)

_, accuracy = sqnl.evaluate_generator(test_generator, nb_samples)

print('accuracy on test set = ',  round((accuracy * 100),2 ), '% ') 


# In[25]:


gen_label_map = test_generator.class_indices
gen_label_map = dict((v,k) for k,v in gen_label_map.items())
print(gen_label_map)


# In[26]:


from sklearn.metrics import classification_report


# In[27]:


preds = sqnl.predict(test_generator, nb_samples)
preds = preds.argmax(1)
preds = [gen_label_map[item] for item in preds]
labels = test_df['category'].to_numpy()

print(classification_report(labels, preds))

