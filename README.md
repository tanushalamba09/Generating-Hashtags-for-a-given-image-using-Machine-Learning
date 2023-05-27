# Generating-Hashtags-for-a-given-image-using-Machine-Learning
#the code is as follows
import tensorflow as tf
import numpy as np
import cv2
import requests


model = tf.keras.applications.ResNet50(weights='imagenet')


def preprocess_image(image):

    image = cv2.resize(image, (224, 224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image / 255.0
 
    image = np.expand_dims(image, axis=0)
    return image


def generate_hashtags(image_path):
    
    image = cv2.imread(image_path)
 
    preprocessed_image = preprocess_image(image)
   
    hashtags = model.predict(preprocessed_image)
    
    top_hashtags_indices = np.argsort(hashtags[0])[::-1][:10]
    top_hashtags = [f'#{index}' for index in top_hashtags_indices]
    return top_hashtags


image_path = r'C:\Users\tanus\Downloads\Screenshot 2023-02-03 at 3.56.22 PM (1).png'
top_hashtags = generate_hashtags(image_path)
print(top_hashtags)
