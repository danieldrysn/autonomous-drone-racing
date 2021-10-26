import numpy as np
import cv2

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

# Define loss functions for model, not used but required to load model
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss     

# Sort labels in clockwise order, starting with the top left corner
def sort_labels(labels):
    sorted_labels = []

    x_sorted = labels[np.argsort(labels[:, 0]), :]
    left_labels = x_sorted[:2, :]
    right_labels = x_sorted[2:, :]

    left_labels = left_labels[np.argsort(left_labels[:, 1]), :]
    sorted_labels.append(left_labels[0])

    right_labels = right_labels[np.argsort(right_labels[:, 1]), :]
    sorted_labels.append(right_labels[0])
    sorted_labels.append(right_labels[1])
    sorted_labels.append(left_labels[1])

    return np.expand_dims(np.array(sorted_labels).ravel(), axis=0).tolist()

# Predict the coordinates of the gate in a given OpenCV image
class GenerateFinalDetections:
    def __init__(self, weights_path):
        self.img_shape = (128, 192, 3)
        self.save_model_path = weights_path
        self.model = models.load_model(self.save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})

    # Generate mask using model, then find center of contours
    def predict(self,img):
        image = cv2.resize(img, (self.img_shape[1], self.img_shape[0]))
        image = image / 255

        image = np.expand_dims(image, axis=0)
        predicted_label = self.model.predict(image)[0]

        mask_img = predicted_label[:, :, 0]
        mask_img = (mask_img / np.amax(mask_img) * 255).astype(np.uint8)

        thresh = cv2.threshold(mask_img, 60, 255, cv2.THRESH_BINARY)[1]

        vals = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = vals[len(vals) - 2]

        if len(contours) != 4:
            return [[]]

        labels = []
        for c in contours:
            M = cv2.moments(c)
 
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"] * 1296 / 192
                cY = M["m01"] / M["m00"] * 864 / 128
            else:   
                return [[]]

            labels.append([cX, cY])

        return sort_labels(np.array(labels))
