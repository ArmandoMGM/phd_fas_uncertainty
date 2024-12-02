import os
import numpy as np
import cv2
import gdown
import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model


class RGBtoHSV(Layer):
    def __init__(self, **kwargs):
        super(RGBtoHSV, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.image.rgb_to_hsv(inputs)

    def get_config(self):
        config = super(RGBtoHSV, self).get_config()
        return config

class RGBtoYCbCr(Layer):
    def __init__(self, **kwargs):
        super(RGBtoYCbCr, self).__init__(**kwargs)

    def call(self, inputs):
        rgb_to_ycbcr_kernel = tf.constant([[0.299, 0.587, 0.114],
                                           [-0.1687, -0.3313, 0.5],
                                           [0.5, -0.4187, -0.0813]])
        offset = tf.constant([0, 128/255, 128/255], dtype=tf.float32)
        ycbcr = tf.tensordot(inputs, rgb_to_ycbcr_kernel, axes=[[3], [1]]) + offset

        return ycbcr

    def get_config(self):
        config = super(RGBtoYCbCr, self).get_config()
        return config

class FuzzyPooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', fuzzy_k=2, **kwargs):
        super(FuzzyPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.upper()
        self.fuzzy_k = fuzzy_k  

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )

        batch_size = tf.shape(inputs)[0]
        new_height = tf.shape(patches)[1]
        new_width = tf.shape(patches)[2]
        channels = inputs.shape[-1]
        patch_dim = self.pool_size[0] * self.pool_size[1]

        patches = tf.reshape(patches, [batch_size, new_height, new_width, channels, patch_dim])

        max_vals = tf.reduce_max(patches, axis=-1, keepdims=True)
        min_vals = tf.reduce_min(patches, axis=-1, keepdims=True)
        denom = max_vals - min_vals + 1e-6 

        membership = 1 - tf.abs(patches - max_vals) / denom
        membership = tf.pow(membership, self.fuzzy_k)

        numerator = tf.reduce_sum(patches * membership, axis=-1)
        denominator = tf.reduce_sum(membership, axis=-1) + 1e-6
        fuzzy_pool = numerator / denominator

        return fuzzy_pool

    def compute_output_shape(self, input_shape):
        if self.padding == 'VALID':
            out_height = (input_shape[1] - self.pool_size[0]) // self.strides[0] + 1
            out_width = (input_shape[2] - self.pool_size[1]) // self.strides[1] + 1
        elif self.padding == 'SAME':
            out_height = (input_shape[1] + self.strides[0] - 1) // self.strides[0]
            out_width = (input_shape[2] + self.strides[1] - 1) // self.strides[1]
        else:
            raise ValueError(f"Invalid padding type: {self.padding}")

        return (input_shape[0], out_height, out_width, input_shape[3])

    def get_config(self):
        config = super(FuzzyPooling, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'fuzzy_k': self.fuzzy_k,
        })
        return config

class GlobalFuzzyPooling2D(Layer):
    def __init__(self, fuzzy_k=2, **kwargs):
        super(GlobalFuzzyPooling2D, self).__init__(**kwargs)
        self.fuzzy_k = fuzzy_k 

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = inputs.shape[3]

        inputs_flat = tf.reshape(inputs, [batch_size, height * width, channels])

        max_vals = tf.reduce_max(inputs_flat, axis=1, keepdims=True) 
        min_vals = tf.reduce_min(inputs_flat, axis=1, keepdims=True)  
        denom = max_vals - min_vals + 1e-6 

        membership = 1 - tf.abs(inputs_flat - max_vals) / denom  
        membership = tf.pow(membership, self.fuzzy_k)

        numerator = tf.reduce_sum(inputs_flat * membership, axis=1)  
        denominator = tf.reduce_sum(membership, axis=1) + 1e-6  
        fuzzy_global_pool = numerator / denominator 

        return fuzzy_global_pool  

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])

    def get_config(self):
        config = super(GlobalFuzzyPooling2D, self).get_config()
        config.update({
            'fuzzy_k': self.fuzzy_k,
        })
        return config

class FASClassifier():
    def __init__(self,  
                 fas_model_name="[CeAS_Full][Intra-Test]ResNet-18-MCD-FP.keras", 
                 fas_model_path="/content/phd_fas_uncertainty/models/",
                 no_times=25):
        """
        Initialize Face Anti-Spoofing model.
        """
        self.img_width = 256
        self.img_height = 256
        self.no_times = no_times

        self.url_drive = 'https://drive.google.com/uc?id=15ogI5e_W8SFxfsy2wthOPoTOlCDe2a3A'

        self.fas_model_name = fas_model_name
        self.fas_model_path = f"{fas_model_path}{self.fas_model_name}"
        self.fas_model = None  
        self.load_models()

    def load_models(self):
        """
        Load Face Anti-Spoofing model.
        """
        if not os.path.exists(self.fas_model_path):
            print(f"File '{self.fas_model_name}' does not exist. Downloading file from drive...")
            os.makedirs(os.path.dirname(self.fas_model_path), exist_ok=True)

            try:
                gdown.download(self.url_drive, self.fas_model_path, quiet=False)
                print(f"File downloaded successfully to: {self.fas_model_path}")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        try:
            self.fas_model = load_model(self.fas_model_path, 
                                        custom_objects={
                                        'RGBtoHSV': RGBtoHSV,
                                        'RGBtoYCbCr': RGBtoYCbCr,
                                        'FuzzyPooling': FuzzyPooling,
                                        'GlobalFuzzyPooling2D': GlobalFuzzyPooling2D})
            
            print(f"Loaded Face Anti-Spoofing Model successfully")
        except Exception as e:
            print(f"Loading Face Anti-Spoofing Model, an unexpected error occurred: {e}")
    
    def preprocess_image(self, img):
        """
        Preprocess image for Face Anti-Spoofing model.
        """
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = np.expand_dims(img, axis=0)/255.0
        return img

    def postprocess_prediction(self, predictions):
        """
        Postprocess prediction for Face Anti-Spoofing model.
        """
        mean_confidence = np.mean(predictions)
        std_confidence = np.std(predictions) 
        label_prediction = 'real'
        return label_prediction, mean_confidence, std_confidence
    
    def predict(self, img):
        """
        Predict Face Anti-Spoofing model.
        """
        img_to_model = self.preprocess_image(img)

        predict_array = None
        for i in tqdm.tqdm(range(self.no_times), total=self.no_times):
            predict = self.fas_model.predict(img_to_model, verbose=False)[-1][0,1:]

            if i==0:
                predict_array = predict.copy()
            else:
                predict_array = np.concatenate([predict_array, predict])

        output = self.postprocess_prediction(predict_array)

        return output


# img = "load image as numpy array in RGB"
# fas_model = FASClassifier()
# fas_results = fas_model.predict(img)
# print(f"""
# FAS Classifier Model: {fas_results}
# """)