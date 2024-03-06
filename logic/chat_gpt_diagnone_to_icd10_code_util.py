import os
import openai
import pandas as pd


class ChatGPTDiagnoneToICD10CodeUtil:
    _instance = None
    df_block = None
    df_category = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatGPTDiagnoneToICD10CodeUtil, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):

        print("Loading all CSV data for ICD10-Codes into cache...")

        self.df_block = pd.read_csv(
            'ressources/df_block.csv')
        self.df_category = pd.read_csv(
            'ressources/df_category_filtered.csv')

        print("...Done! Successfully loaded all CSV data for ICD10-Codes into cache!")
