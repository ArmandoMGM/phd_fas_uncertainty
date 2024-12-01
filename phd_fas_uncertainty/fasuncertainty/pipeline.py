from phd_fas_uncertainty.fasuncertainty.facedetector import FDwithLandmarks
from phd_fas_uncertainty.fasuncertainty.iqdclassifier import IQDClassifier
from phd_fas_uncertainty.fasuncertainty.fasclassifier import FASClassifier


class Pipeline_FAS():
    def __init__(self):
        """
        Initialize Face Anti-Spoofing System.
        """
        print('Face Detection Module...')
        self.facedetection_model = FDwithLandmarks(plot_points_on_faces=False)

        print('Image Quality Assesment Module...')
        self.iqd_model = IQDClassifier()

        print('Face Anti-Spoofing Module...')
        self.fas_model = FASClassifier()

        self.proba_max = 0.64
        self.std_error = 0.9
        

    def predict(self, img):
        faces_found, new_coord_faces = self.facedetection_model.detect(img)
        iq_results, fas_results = None, None
        fas_decision = 'indeterminate'

        if len(faces_found) > 0:
            iq_results = self.iqd_model.predict(faces_found[0])

            if iq_results[0] == 'original' and iq_results[1] >= self.proba_max:
                fas_results = self.fas_model.predict(faces_found[0])

                if fas_results[1] >= 0.64 and fas_results[2] <= self.std_error:
                    fas_decision = 'real'
                else:
                    fas_decision = 'spoof'
            else:
                fas_results = None
                print('Image Quality:', iq_results[0], '/ Confidence:', iq_results[1])
        else:
            print('No face detected')
            
        return faces_found, new_coord_faces, iq_results, fas_results, fas_decision


# img = "load image as numpy array in RGB"
# fas_pipeline = Pipeline_FAS()
# results = fas_pipeline.predict(img)
# print(f"""
# SYSTEM OUTPUTS
# --------------
# Face Coordenates:     {results[1]}
# IQD Classifier Model: {results[2]}
# FAS Classifier Model: {results[3]}
# Final Decision:       {results[4]}
# """)