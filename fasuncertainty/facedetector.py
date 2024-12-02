import os
import requests
import numpy as np
import cv2


def crop_faces_by_landmarks_lbf(landmarks, img=None, plot_point=False):
    """
    Crop faces by landmarks using LBF model
    """
    new_coord_faces = []
    eyes_nose_index  = list(range(27,48))

    frame = img.copy()

    for landmark in landmarks:
        x_eye_min = np.inf
        x_eye_max = 0
        y_nouse_max = 0
        x_min = np.inf
        y_min = np.inf
        x_max = 0
        y_max = 0
        for idx, (x, y) in enumerate(landmark[0]):
            if idx in eyes_nose_index:
                x_eye_min = min(x, x_eye_min)
                x_eye_max = max(x, x_eye_max)
                y_nouse_max = max(y, y_nouse_max)

            x_min = min(x, x_min)
            y_min = min(y, y_min)
            x_max = max(x, x_max)
            y_max = max(y, y_max)

            if plot_point and frame is not None:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 2)

        diff_X_a = np.abs(x_min-x_eye_min)
        diff_X_b = np.abs(x_max-x_eye_max)
        diff_X = min(diff_X_a,diff_X_b)

        diff_Y = np.abs(y_max-y_nouse_max)

        x_new_min = max(0, x_min-diff_X)
        y_new_min = max(0, y_min-diff_Y)
        x_new_max = min(x_max+diff_X, frame.shape[1])
        y_new_max = y_max

        if plot_point and frame is not None:
            cv2.rectangle(frame, (int(x_new_min), int(y_new_min)), (int(x_new_max), int(y_new_max)), (0, 255, 0), 2)

        coord_head = [int(x_new_min), int(y_new_min), int(x_new_max), int(y_new_max)]
        new_coord_faces.append(coord_head)
    return new_coord_faces, frame


class FDwithLandmarks():
    def __init__(self, 
                 landmark_model_name="lbfmodel.yaml", 
                 landmark_model_path="/content/phd_fas_uncertainty/models/",
                 plot_points_on_faces=False):
        """
        Initialize models for face detection and landmark detection
        """
        self.face_detector = None
        self.landmark_detector = None

        self.landmark_model_name = landmark_model_name
        self.landmark_model_path = f"{landmark_model_path}{self.landmark_model_name}"
        self.url_landmark_model = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"

        self.plot_points_on_faces = plot_points_on_faces

        self.img = None

        self.load_models()

    def load_models(self):
        """
        Load models for face detection and landmark detection
        """
        try:
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print(f"Loaded Face Detector Model successfully")
        except Exception as e:
            print(f"Loading Face Detector Model, an unexpected error occurred: {e}")

        if not os.path.exists(self.landmark_model_path):
            print(f"File '{self.landmark_model_name}' does not exist. Downloading file from original repository...")
            os.makedirs(os.path.dirname(self.landmark_model_path), exist_ok=True)

            try:
                response = requests.get(self.url_landmark_model, stream=True)
                response.raise_for_status() 

                with open(self.landmark_model_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

                print(f"File downloaded successfully to: {self.landmark_model_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        try: 
            self.landmark_detector = cv2.face.createFacemarkLBF()
            self.landmark_detector.loadModel(self.landmark_model_path)
            print(f"Loaded Landmark Model successfully")
        except Exception as e:
            print(f"Loading Landmark Model, an unexpected error occurred: {e}")
        
    def predict_faces(self, img):
        """
        Predict faces coordenates in the input images
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        return faces

    def predict_landmarks(self, img, faces):
        """
        Predict landmarks for faces
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, faces_landmarks = self.landmark_detector.fit(gray, np.array(faces))
        return faces_landmarks

    def detect(self, img):
        """
        Predict faces and landmarks
        """
        self.img = img
        faces = self.predict_faces(img)
        if len(faces)>0:
            landmarks = self.predict_landmarks(img, faces)
            new_coord_faces, img_with_point = crop_faces_by_landmarks_lbf(landmarks, img, plot_point=self.plot_points_on_faces)
            self.img = img_with_point
        else:
            new_coord_faces = []
            
        faces_found = []
        for i_nc in new_coord_faces:
            faces_found.append(self.img[i_nc[1]:i_nc[3], i_nc[0]:i_nc[2]])

        return faces_found, new_coord_faces


# img = "load image as numpy array in RGB"
# FaceDetectionModel = FDwithLandmarks(plot_points_on_faces=False)
# faces_found, new_coord_faces = FaceDetectionModel.detect(img)
# print(f"""
# SYSTEM OUTPUTS
# --------------
# Face Number:          {len(faces_found)}
# Face Coordenates:     {new_coord_faces}
# """)