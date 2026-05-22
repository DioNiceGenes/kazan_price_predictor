import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
from pathlib import Path

from config import settings
from model_training import ModelTrainer
from preprocessing import load_processed_data, create_features



class TextExperiments:
    def __init__(self, X: pd.DataFrame, y: pd.Series, descriptions: pd.Series):
        """
        X — DataFrame после feature engineering (с числовыми и категориальными признаками)
        y — целевая переменная (price)
        descriptions — Series с описаниями
        """
        self.X = X.copy()
        self.y = y.copy()
        
        self.descriptions = descriptions.fillna('').astype(str).copy()
        
        print(f"TextExperiments инициализирован:")
        print(f"   X.shape          = {self.X.shape}")
        print(f"   y.shape          = {self.y.shape}")
        print(f"   descriptions aligned = {self.descriptions.shape}")
        
        self.results = {}
        self.figures_dir = Path("reports/figures/text_experiments")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def create_handcrafted_features(self):
        """Hand-crafted признаки из описания"""
        print("Создание hand-crafted текстовых признаков...")
        
        def extract(text):
            if not text or len(text.strip()) == 0:
                return {k: 0 for k in [
                    'desc_len_chars', 'desc_len_words', 'excl_marks', 'question_marks',
                    'has_euroremont', 'has_designer', 'has_cosmetic', 'has_needs_repair',
                    'has_new_building', 'has_furniture', 'has_view', 'has_river_view',
                    'has_balcony', 'has_loggia', 'has_wardrobe', 'mentions_kitchen',
                    'mentions_bathroom'
                ]}
            
            t = str(text).lower()
            return {
                'desc_len_chars': len(text),
                'desc_len_words': len(text.split()),
                'excl_marks': text.count('!'),
                'question_marks': text.count('?'),
                'has_euroremont': int(any(x in t for x in ['евроремонт', 'евро ремонт', 'евро-ремонт'])),
                'has_designer': int(any(x in t for x in ['дизайнерский', 'авторский'])),
                'has_cosmetic': int('косметический' in t),
                'has_needs_repair': int(any(x in t for x in ['требует ремонта', 'нуждается в ремонте', 'старый ремонт'])),
                'has_new_building': int(any(x in t for x in ['новостройк', 'новый дом', 'сдан'])),
                'has_furniture': int(any(x in t for x in ['мебель', 'меблирован', 'с мебелью'])),
                'has_view': int('вид на' in t),
                'has_river_view': int(any(x in t for x in ['вид на реку', 'вид на волгу', 'вид на казанку'])),
                'has_balcony': int('балкон' in t),
                'has_loggia': int('лоджи' in t),
                'has_wardrobe': int(any(x in t for x in ['шкаф', 'гардероб'])),
                'mentions_kitchen': int(any(x in t for x in ['кухн', 'кухон'])),
                'mentions_bathroom': int(any(x in t for x in ['санузел', 'ванн', 'туалет'])),
            }
        
        return pd.DataFrame([extract(text) for text in self.descriptions], index=self.descriptions.index)

    def create_tfidf_features(self, max_features=1000):
        """TF-IDF"""
        print(f"Создание TF-IDF признаков (max_features={max_features})...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.85,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.descriptions)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
            index=self.descriptions.index
        )
        return tfidf_df, vectorizer

    def run_all_experiments(self):
        """Запуск всех экспериментов"""
        print("\n" + "="*90)
        print("ЭКСПЕРИМЕНТЫ С ТЕКСТОВЫМИ ПРИЗНАКАМИ")
        print("="*90)
        
        trainer = ModelTrainer(self.X, self.y, None, None)
        
        # 1. Baseline
        print("\n1. Baseline — только табличные признаки")
        mae_base, r2_base, _ = trainer.evaluate_xgboost(
            run_name="01_Baseline_Tabular_Only",
            tags={"text": "none"}
        )
        self.results["Baseline (Табличные)"] = {"MAE": mae_base, "R2": r2_base}

        # 2. Hand-crafted
        print("\n2. Hand-crafted признаки")
        handcrafted = self.create_handcrafted_features()
        handcrafted.to_csv(settings.PROCESSED_DIR + "/handcrafted_features.csv")
        X_hc = pd.concat([self.X.reset_index(drop=True), handcrafted.reset_index(drop=True)], axis=1)
        
        mae_hc, r2_hc, _ = trainer.evaluate_xgboost(
            X=X_hc, y=self.y,
            run_name="02_Handcrafted_Features",
            tags={"text": "handcrafted"}
        )
        self.results["Handcrafted Features"] = {"MAE": mae_hc, "R2": r2_hc}

        # # 3. TF-IDF
        # print("\n3. TF-IDF признаки")
        # tfidf_df, _ = self.create_tfidf_features(max_features=1000)
        # X_tfidf = pd.concat([self.X.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        
        # mae_tfidf, r2_tfidf, _ = trainer.evaluate_xgboost(
        #     X=X_tfidf, y=self.y,
        #     run_name="03_TFIDF_1000",
        #     tags={"text": "tfidf"}
        # )
        # self.results["TF-IDF (1000)"] = {"MAE": mae_tfidf, "R2": r2_tfidf}

        # # 4. Полный вариант
        # print("\n4. Полный текст (Handcrafted + TF-IDF)")
        # X_full = pd.concat([X_hc, tfidf_df], axis=1)
        
        # mae_full, r2_full, _ = trainer.evaluate_xgboost(
        #     X=X_full, y=self.y,
        #     run_name="04_Full_Text_Handcrafted+TFIDF",
        #     tags={"text": "full"}
        # )
        # self.results["Full Text"] = {"MAE": mae_full, "R2": r2_full}

        comparison = pd.DataFrame(self.results).T
        comparison = comparison.sort_values("MAE")
        
        print("\n" + "="*90)
        print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
        print("="*90)
        print(comparison.round(4))
        
        comparison.to_csv(settings.ALL_DATA_PATH + "/reports/text_features_comparison.csv")
        
        return comparison


if __name__ == "__main__":
    # Загружаем данные
    tdf = pd.read_csv(settings.TDF_PATH)
    descriptions = pd.read_csv(settings.DESCRIPTIONS_PATH)['description']
    
    # Создаём X, y
    data_dict = create_features(tdf)
    
    preprocessor = data_dict["preprocessor"]
    X = pd.DataFrame(preprocessor.transform(data_dict['X']), columns=data_dict['feature_names'], index=data_dict["X"].index)
    y = data_dict["y"]

    # Запускаем эксперименты
    experiments = TextExperiments(
        X=X,
        y=y, 
        descriptions=descriptions,
    )
    
    comparison = experiments.run_all_experiments()
