import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import time


def GetTensorflowVersion():
    print(tf.__version__)


class TFModel:
    def __init__(self, class_names, model_path):
        GetTensorflowVersion()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.class_names = class_names

    def predict_img(self, image_url, num_classes=1):

        print("Predicting image...", image_url, "- num_classes:", num_classes)
        start_time = time.time()
        img = keras.preprocessing.image.load_img(image_url,
                                                 target_size=(500, 500))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.array([img_array])
        predictions = self.model.predict(img_array)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        top_classes_indices = np.argsort(predictions)[0, ::-1][:num_classes]
        top_classes = [self.class_names[i] for i in top_classes_indices]
        top_confidences = [predictions[0, i] for i in top_classes_indices]

        result = {class_name: confidence for class_name,
                  confidence in zip(top_classes, top_confidences)}

        return result, round(total_time)
