import openai
import joblib
from tensorflow.keras.models import load_model


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

    def get_icd10_code_for_diagnose(self, diagnose_text):
        return self._execute_diagnose_translation_for_icd_10_code(diagnose_text)

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

    def _execute_diagnose_translation_for_icd_10_code(self, diagnose_text):
        # Set your OpenAI API key
        openai.api_key = self.__GPTapi_key

        # Convert df_block DataFrame to a list of tuples for easier handling in the prompt
        chapters_info = self.df_block[['Label Text', 'Class Code']].apply(tuple, axis=1).tolist()

        # Create the prompt
        prompt = f"ich gebe dir einen Diagnosetext und ich gebe dir ein Inhaltsverzeichnis in Form von Label Text und Class Codes. Suche mir den Class Code raus, in dem Du den Diagnosetext vermutest (Höchste Wahrscheinlichkeit). Ich will nur den Class Code. Kein weiteres Bla bla. Examplarische Antwort: A00-A09. DiagnoseText: {diagnose_text}. Basierend auf dem Inhaltsverzeichnis: {chapters_info}.Wenn du nichts findest, dann gebe diese Antwort wieder: Dies ist ein reduziertes Sprachmodell im Rahmen der OptaData-Challenge und kann daher dein Vorschlag nicht finden."

        # Run model
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Use the response to filter df_category
        response_code = response.choices[0].message['content'].strip().replace('"', '').replace("'",
                                                                                                "")  # Strip both double and single quotation marks
        filtered_df_category = self.df_category[
            self.df_category['Block of Category'].str.contains(response_code)].copy()

        # Convert the filtered DataFrame to a list of tuples for the second prompt
        filtered_chapters_info = filtered_df_category[['Label Text', 'Class Code']].apply(tuple, axis=1).tolist()

        # Create the second prompt with the filtered DataFrame
        prompt_2 = f"ich gebe dir einen Diagnosetext und ich gebe dir ein Inhaltsverzeichnis in Form von Label Text und Class Codes. Suche mir den Class Code raus, in dem Du den Diagnosetext vermutest (höchste Wahrscheinlichkeit). Ich will nur den Class Code. Kein weiteres Bla bla. Examplarische Antwort: A00. DiagnoseText: {diagnose_text}. Durchsuche die Liste: {filtered_chapters_info}.Gib mir den Class Code wieder, durchsuche den Label Text. Wenn du nichts findest, dann gebe diese Antwort wieder: Dies ist ein reduziertes Sprachmodell im Rahmen der OptaData-Challenge und kann daher dein Vorschlag nicht finden."

        # Run second model
        response_2 = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_2}
            ]
        )

        # Return the second GPT-3 response, stripping both double and single quotation marks
        final_response = response_2.choices[0].message['content'].strip().replace('"', '').replace("'", "")
        return final_response
