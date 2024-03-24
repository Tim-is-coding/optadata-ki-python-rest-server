import openai
import pandas as pd


class ChatGPTDiagnoneToICD10CodeUtil:
    _instance = None
    df_block = None
    df_category = None
    __GPTapi_key = "PUT_YOUR_OPENAI_API_KEY_HERE"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatGPTDiagnoneToICD10CodeUtil, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def get_icd10_code_for_diagnose(self, diagnose_text):
        return self._execute_diagnose_translation_for_icd_10_code(diagnose_text)

    def _load_data(self):
        print("Loading all CSV data for ICD10-Codes into cache...")

        self.df_block = pd.read_csv(
            'data/csv/df_block.csv')
        self.df_category = pd.read_csv(
            'data/csv/df_category_filtered.csv')

        print("...Done! Successfully loaded all CSV data for ICD10-Codes into cache!")

    def _execute_diagnose_translation_for_icd_10_code(self, diagnoseText):
        # Set your OpenAI API key
        openai.api_key = self.__GPTapi_key

        # Convert df_block DataFrame to a list of tuples for easier handling in the prompt
        chapters_info = self.df_block[['Label Text', 'Class Code']].apply(tuple, axis=1).tolist()

        # Create the prompt
        prompt = f"ich gebe dir einen Diagnosetext und ich gebe dir ein Inhaltsverzeichnis in Form von Label Text und Class Codes. Suche mir den Class Code raus, in dem Du den Diagnosetext vermutest (Höchste Wahrscheinlichkeit). Ich will nur den Class Code. Kein weiteres Bla bla. Examplarische Antwort: A00-A09. DiagnoseText: {diagnoseText}. Basierend auf dem Inhaltsverzeichnis: {chapters_info}.Wenn du nichts findest, dann gebe diese Antwort wieder: Dies ist ein reduziertes Sprachmodell im Rahmen der OptaData-Challenge und kann daher dein Vorschlag nicht finden."

        # Run model
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
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
        prompt_2 = f"ich gebe dir einen Diagnosetext und ich gebe dir ein Inhaltsverzeichnis in Form von Label Text und Class Codes. Suche mir den Class Code raus, in dem Du den Diagnosetext vermutest (höchste Wahrscheinlichkeit). Ich will nur den Class Code. Kein weiteres Bla bla. Examplarische Antwort: A00. DiagnoseText: {diagnoseText}. Durchsuche die Liste: {filtered_chapters_info}.Gib mir den Class Code wieder, durchsuche den Label Text. Wenn du nichts findest, dann gebe diese Antwort wieder: Dies ist ein reduziertes Sprachmodell im Rahmen der OptaData-Challenge und kann daher dein Vorschlag nicht finden."

        # Run second model
        response_2 = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_2}
            ]
        )

        # Return the second GPT-3 response, stripping both double and single quotation marks
        final_response = response_2.choices[0].message['content'].strip().replace('"', '').replace("'", "")
        return final_response
