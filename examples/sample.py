
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from time import time

import tensorflow as tf

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)
#
#keras.backend.tensorflow_backend.set_session(get_session())

#model_path = 'D:\\Abhishek\\our_model\\dataset\\snapshots\\model_new.h5'    ## replace this with your model path
model_path = r"E:\IISc\Object_detection\keras-retinanet\keras-retinanet-main\snapshots\resnet50_csv_47_converted.h5"   #
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'bicycle', 1: 'bus', 2: 'traffic sign', 3: 'motorcycle', 4: 'car', 5: 'traffic light', 6: 'person', 7: 'vehicle fallback', 8: 'truck', 9: 'autorickshaw', 10: 'animal', 11: 'rider'}


#image_path = 'D:\\Abhishek\\our_model\\dataset\\JPEGImages\\89_509098_leftImg8bit.jpeg'  ## replace with input image path
image_path = r"E:\IISc\Object_detection\IDD\IDD_Detection\test\images\0_005506_leftImg8bit.jpeg"

# output_path = 'D:\\Abhishek\\ImageAI_demopurpose\\80.jpg'   ## replace with output image path

# output_path = "E:\keras-retinanet\keras-retinanet-main"

def detection_on_image(image_path):

        image = cv2.imread(image_path)

        draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        time1 = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        time2 = time.time()
        cal_time = time2 - time1
        boxes /= scale
        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            if score < 0.3:
                break

            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            print(labels_to_names[label],box,score)
            draw_caption(draw, b, caption)
        detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
#        cv2.imwrite(output_path, detected_img)
        cv2.imshow('Detection',detected_img)
        cv2.waitKey(0)
detection_on_image(image_path)



#run this script   are you there?
