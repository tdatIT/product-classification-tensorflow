import numpy as np
import pandas as pd
import tensorflow as tf
import keras


def GetTensorflowVersion():
    print(tf.__version__)


class TFModel:
    def __init__(self, class_names, model_path):
        GetTensorflowVersion()
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names

    def predict_img(self, image_url):
        img = keras.preprocessing.image.load_img(image_url,
                                                 target_size=(500, 500))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.array([img_array])
        predictions = self.model.predict(img_array)
        class_id = np.argmax(predictions, axis=1)
        return self.class_names[class_id.item()]
