#!/usr/bin/env python
# coding: utf-8

# Load necessary modules

import sys
sys.path.insert(0, '../')


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = "0,"

# set the modified tf session as backend in keras
# setup_gpu(gpu)


# ## Load RetinaNet model

# In[ ]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_csv_47_converted.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'bicycle', 1: 'bus', 2: 'traffic sign', 3: 'motorcycle', 4: 'car', 5: 'traffic light', 6: 'person', 7: 'vehicle fallback', 8: 'truck', 9: 'autorickshaw', 10: 'animal', 11: 'rider'}


# ## Run detection on example


# load image
image = read_image_bgr('E:\\IISc\\Object_detection\\IDD\\IDD_Detection\\test\\images\\1_741238_leftImg8bit.jpeg')

#frontFar_BLR-2018-03-22_17-39-26_2_frontFar_0000060

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# print(scores)


print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale
print(boxes)
print(scores)
print(labels)
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    color = label_color(label)

    b = box.astype(int)
    # print("box indi\n",b)
    draw_box(draw, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()


