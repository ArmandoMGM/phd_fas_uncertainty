import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model


class IQDClassifier():
    def __init__(self,  
                 iqd_model_name="[FAS-IQA]MobileNetV2.keras", 
                 iqd_model_path="/content/phd_fas_uncertainty/models/"):
        """
        Initialize image quality distortion classifier model for face anti-spoofing systems.
        """
        self.img_width = 256
        self.img_height = 256

        self.iqd_dictionary = { 
            0: 'BlurXY_11',
            1: 'BlurXY_15',
            2: 'BlurXY_7',
            3: 'BlurX_11',
            4: 'BlurX_15',
            5: 'BlurX_7',
            6: 'BlurY_11',
            7: 'BlurY_15',
            8: 'BlurY_7',
            9: 'GaussianBlur_11',
            10: 'GaussianBlur_15',
            11: 'GaussianBlur_7',
            12: 'hbright_10',
            13: 'hbright_5',
            14: 'hbright_8',
            15: 'jpgcompression_10',
            16: 'jpgcompression_30',
            17: 'jpgcompression_50',
            18: 'lbright_0.1',
            19: 'lbright_0.2',
            20: 'lbright_0.3',
            21: 'noise_15',
            22: 'noise_25',
            23: 'noise_45',
            24: 'original'
            }

        self.iqd_model_name = iqd_model_name
        self.iqd_model_path = f"{iqd_model_path}{self.iqd_model_name}"
        self.iqd_model = None        

        self.load_model()
        
    def load_model(self):
        """
        Load image quality distortion classifier model
        """
        try:
            self.iqd_model = load_model(self.iqd_model_path)
            print(f"Loaded IQD Classifier Model successfully")
        except Exception as e:
            print(f"Loading IQD Classifier Model, an unexpected error occurred: {e}")
    
    def preprocess_image(self, img):
        """
        Preprocess image for image quality distortion classifier model
        """
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = np.expand_dims(img, axis=0)/255.0
        return img

    def postprocess_prediction(self, prediction):
        """
        Postprocess prediction for image quality distortion classifier model
        """
        index_prediction = np.argmax(prediction)
        confidence = np.max(prediction)
        label_prediction = self.iqd_dictionary[index_prediction]
        return label_prediction, confidence
    
    def predict(self, img):
        """
        Predict image quality distortion
        """
        img_to_model = self.preprocess_image(img)

        prediction = self.iqd_model.predict(img_to_model, verbose=False)

        output = self.postprocess_prediction(prediction)

        return output


# img = "load image as numpy array in RGB"
# iqd_model = IQDClassifier()
# iq_results = iqd_model.predict(faces_found[0])
# print(f"""
# IQD Classifier Model: {iq_results}
# """)