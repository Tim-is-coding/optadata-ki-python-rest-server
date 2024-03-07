import threading

import joblib
from tensorflow.keras.models import load_model
import numpy as np

from model.ai_recommondation import AiRecommondation
from model.ai_recommondation_item import AiRecommondationItem
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

    # threading
    lock = threading.Lock()
    threads = []

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

        recommondations = self._execute_recommondation_models(icd_10_code, krankenkassen_ik, bundesland)

        # sort recommondations by probability
        recommondations.sort(key=lambda x: x.probability, reverse=True)
        return recommondations

    def _execute_recommondation_models(self, icd_10_code, krankenkassen_ik, bundesland):
        predictions = []

        # step 1: predict product
        recommendations, probabilities = self._predict_product(icd_10_code, krankenkassen_ik, bundesland)

        # step 2: predict price
        for i in range(len(recommendations)):
            recommondation = recommendations[i]
            probability = probabilities[i]

            thread = threading.Thread(target=self._predict_price_thread_safe,
                                      args=(icd_10_code, krankenkassen_ik, bundesland,
                                            recommondation,
                                            probability, predictions))
            thread.start()
            self.threads.append(thread)

        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()

        return predictions

    def _predict_price_thread_safe(self, icd_10_code, krankenkassen_ik, bundesland, recommondation, probability,
                                   predictions):

        # call the price prediction model for each recommendation
        price_options = self._predict_price(icd_10_code, krankenkassen_ik, bundesland, recommondation)

        prices = []
        for price_option in price_options:
            price = price_option[0]
            price_probability = price_option[1]

            price_option_as_bean = AiRecommondationItem(value=str(price), percentage=int(price_probability * 100))
            prices.append(price_option_as_bean)

        hilfsmittelnummer_option_as_bean = AiRecommondationItem(value=recommondation,
                                                                percentage=int(probability * 100))

        ai_recommondation = AiRecommondation(hilfsmittelNummer=hilfsmittelnummer_option_as_bean,
                                             prices=prices,
                                             probability=int(probability * 100))

        with self.lock:  # Ensure thread-safe append operation
            predictions.append(ai_recommondation)

    def _predict_product(self, icd_10_code, krankenkassen_ik, bundesland):
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

        return recommendations, probabilities

    def _predict_price(self, icd_10_code, krankenkassen_ik, bundesland, hilfsmittel_nummer):
        icd_10_code_encoded = self.le_icd10.transform([icd_10_code])
        krankenkassen_ik_encoded = self.le_insurance.transform([krankenkassen_ik])
        positionsnummer_encoded = self.le_positionsnummer.transform([hilfsmittel_nummer])

        bundesland_zeros = np.zeros((1, len(self.bundesland_mapping)))
        if bundesland != 'Brandenburg':
            bundesland_index = self.bundesland_mapping.index('Bundesland_' + bundesland)
            bundesland_zeros[0, bundesland_index] = 1

        icd10_code = np.array([icd_10_code_encoded]).reshape(1, -1)
        insurance_id = np.array([krankenkassen_ik_encoded]).reshape(1, -1)
        positionsnummer = np.array([positionsnummer_encoded]).reshape(1, -1)

        # Run model prediction
        price_probabilities = self.model_price_adjusted.predict(
            [icd10_code, insurance_id, positionsnummer, bundesland_zeros])

        # Extract the top 3 price indices based on the highest probabilities
        top_3_price_indices = np.argsort(price_probabilities[0])[-3:][::-1]

        price_options = []
        for idx in top_3_price_indices:
            predicted_price = self.le_preis.inverse_transform([idx])[0]
            probability = price_probabilities[0][idx]
            price_options.append((predicted_price, probability))

        return price_options

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
