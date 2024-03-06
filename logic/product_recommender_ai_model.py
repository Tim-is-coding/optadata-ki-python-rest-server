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
