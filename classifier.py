import mxnet as mx
import gluoncv as gcv 
import matplotlib.pyplot as plt
import numpy as np
import os

CLASS_MODEL = './models/'

# Function to load the image
def load_image(path):
    image = mx.image.imread(path)
    return image

# Funtion to load a pretrained model
def load_model():
    network = gcv.model_zoo.get_model('MobileNet1.0',root=CLASS_MODEL,pretrained=True)
    return network

# Funtion to Transform and Batch an image
def transform_image(image):
    transformed_image = gcv.data.transforms.presets.imagenet.transform_eval(image)
    return transformed_image

# Function to predict the probability of various classes
def predict_probability(network,image):
    prediction = network(image)
    prediction = prediction[0]
    probability = mx.nd.softmax(prediction)

    return probability

# Function to return the Class index for a particular Class name
def find_class_idx(label):
    for i in network.classes:
        if i == label:
            return network.classes.index(i)

# Funtion to slice out Tennis Ball class
def slice_tennis_ball_class(pred_probas):
    class_index = find_class_idx('tennis ball')
    class_prob = pred_probas[class_index]
    return class_prob.asscalar()


image_url_1 = './Images/football.jpeg'
image_url_2 = './Images/tennis_balls.jpeg'

# For First Image
image = load_image(image_url_1)
network = load_model()
transform = transform_image(image)
prob = predict_probability(network,transform)
a = slice_tennis_ball_class(prob)
print('Image 1 ---> {0:.2%} Confidence that it is a Tennis Ball'.format(a))


# For Second Image
image = load_image(image_url_2)
network = load_model()
transform = transform_image(image)
prob = predict_probability(network,transform)
a = slice_tennis_ball_class(prob)
print('Image 2 ---> {0:.2%} Confidence that it is a Tennis Ball'.format(a))
