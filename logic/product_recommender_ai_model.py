import joblib
from tensorflow.keras.models import load_model
import numpy as np

from model.recommondation_request import RecommendationRequest


class ProductRemcommenderAiModel:
    _instance = None

    # label encoders
    le_icd10 = None
    le_insurance = None
    le_positionsnummer = None
    le_preis = None
    bundesland_mapping = None

    # ai models
    model_product_adjusted = None
    model_price_adjusted = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProductRemcommenderAiModel, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def recommend_hilfsmittel(self, recommondation_request: RecommendationRequest):
        icd_10_code = recommondation_request.icd10Code
        krankenkassen_ik = recommondation_request.krankenkassenIk
        bundesland = recommondation_request.bundesLand
        assert icd_10_code is not None
        assert krankenkassen_ik is not None
        assert bundesland is not None

        return self._execute_recommondation_models(icd_10_code, krankenkassen_ik, bundesland)

    def _execute_recommondation_models(self, icd_10_code, krankenkassen_ik, bundesland):

        # step 1: predict product
        icd_10_code_encoded = self.le_icd10.transform([icd_10_code])
        krankenkassen_ik_encoded = self.le_insurance.transform([krankenkassen_ik])

        bundesland_zeros = np.zeros((1, len(self.bundesland_mapping)))
        if bundesland != 'Brandenburg':
            bundesland_index = self.bundesland_mapping.index('Bundesland_' + bundesland)
            bundesland_zeros[0, bundesland_index] = 1

        icd10_code = np.array([icd_10_code_encoded]).reshape(1, -1)
        insurance_id = np.array([krankenkassen_ik_encoded]).reshape(1, -1)

        # Run model prediction
        predictions = self.model_product_adjusted.predict([icd10_code, insurance_id, bundesland_zeros])

        # Get top 3 predictions and probabilities
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = np.sort(predictions[0])[-3:][::-1]

        recommendations = self.le_positionsnummer.inverse_transform(top_3_indices)
        probabilities = top_3_probs

        # Print recommendations and probabilities
        print("Top 3 Product Article Numbers and Probabilities:")
        for product, prob in zip(recommendations, probabilities):
            print(f'Product Article Number: {product}, Probability: {prob:.4f}')


    def _predict_product(self, icd_10_code_encoded, krankenkassen_ik_encoded, bundesland_encoded):
        # predict product
        product_prediction = self.model_product_adjusted.predict([icd_10_code_encoded, krankenkassen_ik_encoded, bundesland_encoded])
        return product_prediction

    def _load_data(self):
        print("Loading all Label Encoders and H5 models...")

        encoder_path = 'data/joblib/'

        self.le_icd10 = joblib.load(encoder_path + 'le_icd10.joblib')
        self.le_insurance = joblib.load(encoder_path + 'le_insurance.joblib')
        self.le_positionsnummer = joblib.load(encoder_path + 'le_positionsnummer.joblib')
        self.le_preis = joblib.load(encoder_path + 'le_preis.joblib')
        self.bundesland_mapping = joblib.load(encoder_path + 'bundesland_mapping.joblib')

        self.model_product_adjusted = load_model('data/h5/M2V1_ProductRecommendation_selbstlernend.h5')
        self.model_price_adjusted = load_model('data/h5/M2V2_PriceRecommendation_selbstlernend.h5')

        print("...Done! Successfully loaded all Label Encoders and H5 models!")
